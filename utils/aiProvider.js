const DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite";

let fetchFn = globalThis.fetch;
if (!fetchFn) {
  try {
    const nodeFetch = require("node-fetch");
    fetchFn = nodeFetch.default || nodeFetch;
  } catch {
    // If neither global fetch nor node-fetch exists, callers will get a clear error at runtime.
    fetchFn = null;
  }
}

const fetch = fetchFn ? fetchFn.bind(globalThis) : null;

function normalizeProviderName(provider) {
  if (!provider) return null;
  const normalized = String(provider).trim().toLowerCase();
  if (normalized === "gemini" || normalized === "google") return "gemini";
  if (
    normalized === "openai" ||
    normalized === "openai-compat" ||
    normalized === "openai_compat" ||
    normalized === "openai-compatible" ||
    normalized === "openrouter" ||
    normalized === "zai"
  ) {
    return "openai-compat";
  }
  return null;
}

function validateExternalUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error(`Invalid URL: ${rawUrl}`);
  }

  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error(`Disallowed URL scheme: ${parsed.protocol}`);
  }

  const hostname = parsed.hostname.toLowerCase();

  // Block loopback
  if (hostname === "localhost" || hostname === "::1") {
    throw new Error("Requests to loopback addresses are not allowed");
  }

  // Parse IPv4
  const ipv4 = hostname.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (ipv4) {
    const [, a, b, c] = ipv4.map(Number);
    if (
      a === 127 || // loopback
      a === 10 || // RFC 1918
      (a === 172 && b >= 16 && b <= 31) || // RFC 1918
      (a === 192 && b === 168) || // RFC 1918
      (a === 169 && b === 254) || // link-local / cloud metadata
      a === 0 // reserved
    ) {
      throw new Error("Requests to private/reserved IP ranges are not allowed");
    }
  }
}

function getOpenAIChatCompletionsUrl(baseUrl) {
  const raw = (baseUrl || "").trim().replace(/\/+$/, "");
  if (!raw) return "https://api.openai.com/v1/chat/completions";
  if (raw.includes("/chat/completions")) return raw;
  if (raw.endsWith("/v1")) return `${raw}/chat/completions`;
  return `${raw}/v1/chat/completions`;
}

function normalizeTemperature(value) {
  if (value === undefined || value === null || value === "") return undefined;
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(num)) return undefined;
  return Math.max(0, Math.min(1, num));
}

function parseOptionalJsonObject(text) {
  if (!text) return null;
  const raw = String(text).trim();
  if (!raw) return null;
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Extra headers must be a JSON object");
  }
  return parsed;
}

function buildExtraHeaders(extraHeaders) {
  if (!extraHeaders) return {};

  const forbidden = new Set(["authorization", "content-type", "content-length", "host"]);
  const headers = {};

  for (const [key, value] of Object.entries(extraHeaders)) {
    const headerName = String(key).trim();
    if (!headerName) continue;
    if (forbidden.has(headerName.toLowerCase())) continue;
    if (value === undefined || value === null) continue;
    headers[headerName] = String(value).replace(/[\r\n]/g, "");
  }

  return headers;
}

function getAiProviderConfigFromConfig(configData = {}) {
  const provider = normalizeProviderName(configData.AiProvider);
  const temperature = normalizeTemperature(configData.AiTemperature);

  if (provider === "openai-compat") {
    return {
      provider: "openai-compat",
      apiKey: (configData.OpenAICompatApiKey || "").trim(),
      baseUrl: (configData.OpenAICompatBaseUrl || "").trim(),
      model: (configData.OpenAICompatModel || "gpt-4o-mini").trim(),
      extraHeaders: (configData.OpenAICompatExtraHeaders || "").trim(),
      timeoutMs: Number(configData.OpenAICompatTimeoutMs) || undefined,
      temperature,
    };
  }

  if (provider === "gemini") {
    return {
      provider: "gemini",
      apiKey: (configData.GeminiApiKey || "").trim(),
      model: (configData.GeminiModel || DEFAULT_GEMINI_MODEL).trim(),
      temperature,
      enableGrounding: configData.EnableGeminiGrounding === true,
    };
  }

  // Backwards compatibility: older configs only had Gemini fields.
  const hasGeminiKey = !!(configData.GeminiApiKey && String(configData.GeminiApiKey).trim());
  const hasOpenAICompatKey = !!(
    configData.OpenAICompatApiKey && String(configData.OpenAICompatApiKey).trim()
  );

  if (hasOpenAICompatKey && !hasGeminiKey) {
    return {
      provider: "openai-compat",
      apiKey: String(configData.OpenAICompatApiKey).trim(),
      baseUrl: (configData.OpenAICompatBaseUrl || "").trim(),
      model: (configData.OpenAICompatModel || "gpt-4o-mini").trim(),
      extraHeaders: (configData.OpenAICompatExtraHeaders || "").trim(),
      timeoutMs: Number(configData.OpenAICompatTimeoutMs) || undefined,
      temperature,
    };
  }

  return {
    provider: "gemini",
    apiKey: (configData.GeminiApiKey || "").trim(),
    model: (configData.GeminiModel || DEFAULT_GEMINI_MODEL).trim(),
    temperature,
    enableGrounding: configData.EnableGeminiGrounding === true,
  };
}

function createAiTextGenerator(aiProviderConfig) {
  if (!aiProviderConfig || !aiProviderConfig.provider) {
    throw new Error("AI provider configuration is missing");
  }

  if (aiProviderConfig.provider === "gemini") {
    return {
      provider: "gemini",
      model: aiProviderConfig.model,
      async generateText(prompt) {
        const { GoogleGenerativeAI } = require("@google/generative-ai");
        const genAI = new GoogleGenerativeAI(aiProviderConfig.apiKey);
        const model = genAI.getGenerativeModel({
          model: aiProviderConfig.model,
          generationConfig: {
            temperature:
              typeof aiProviderConfig.temperature === "number"
                ? aiProviderConfig.temperature
                : 0.2,
          },
          ...(aiProviderConfig.enableGrounding ? { tools: [{ googleSearch: {} }] } : {}),
        });
        const aiResult = await model.generateContent(prompt);
        return aiResult.response.text().trim();
      },
    };
  }

  if (aiProviderConfig.provider === "openai-compat") {
    return {
      provider: "openai-compat",
      model: aiProviderConfig.model,
      async generateText(prompt) {
        if (!fetch) {
          throw new Error(
            "Fetch API is not available (need Node 18+ or install node-fetch)"
          );
        }

        const timeoutMs =
          typeof aiProviderConfig.timeoutMs === "number" && aiProviderConfig.timeoutMs > 0
            ? aiProviderConfig.timeoutMs
            : 30000;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        let extraHeadersObj;
        try {
          extraHeadersObj = parseOptionalJsonObject(aiProviderConfig.extraHeaders);
        } catch (error) {
          const parseError = new Error(`Invalid extra headers JSON: ${error.message}`);
          parseError.status = 400;
          throw parseError;
        }

        const url = getOpenAIChatCompletionsUrl(aiProviderConfig.baseUrl);
        validateExternalUrl(url);
        let response;
        try {
          response = await fetch(url, {
            method: "POST",
            signal: controller.signal,
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${aiProviderConfig.apiKey}`,
              ...buildExtraHeaders(extraHeadersObj),
            },
            body: JSON.stringify({
              model: aiProviderConfig.model,
              messages: [{ role: "user", content: prompt }],
              temperature:
                typeof aiProviderConfig.temperature === "number"
                  ? aiProviderConfig.temperature
                  : 0.2,
              max_tokens: 2000,
            }),
          });
        } catch (error) {
          if (error && error.name === "AbortError") {
            const timeoutError = new Error(
              `OpenAI-compatible API request timed out after ${timeoutMs}ms`
            );
            timeoutError.status = 504;
            throw timeoutError;
          }
          throw error;
        } finally {
          clearTimeout(timeoutId);
        }

        if (!response.ok) {
          const errorText = await response.text().catch(() => "");
          const error = new Error(
            `OpenAI-compatible API error (Status: ${response.status})${errorText ? `: ${errorText}` : ""}`
          );
          error.status = response.status;
          throw error;
        }

        const data = await response.json();
        const content =
          data?.choices?.[0]?.message?.content ??
          data?.choices?.[0]?.text ??
          "";

        return String(content).trim();
      },
    };
  }

  throw new Error(`Unsupported AI provider: ${aiProviderConfig.provider}`);
}

module.exports = {
  createAiTextGenerator,
  getAiProviderConfigFromConfig,
  getOpenAIChatCompletionsUrl,
  DEFAULT_GEMINI_MODEL,
};
