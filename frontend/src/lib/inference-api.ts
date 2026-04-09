/**
 * FastAPI cognitive-load inference (backend/main.py).
 * Dev: Vite proxies `/inference` → INFERENCE_PROXY_TARGET (default http://127.0.0.1:8000).
 * Prod / direct: set VITE_INFERENCE_API_URL to the full API origin (e.g. https://api.example.com).
 */

export type PredictResponse = {
	regression_score: number;
	classification_label: string;
	classification_class_index: number;
	probabilities: Record<string, number>;
	n_frames_used: number;
	warnings?: string[];
};

export function getInferenceApiBase(): string {
	const raw = import.meta.env.VITE_INFERENCE_API_URL?.replace(/\/$/, "");
	if (raw) return raw;
	return "/inference";
}

function normalizeDetail(detail: unknown): string {
	if (typeof detail === "string") return detail;
	if (Array.isArray(detail)) {
		return detail
			.map((x) => (typeof x === "object" && x && "msg" in x ? String((x as { msg: string }).msg) : String(x)))
			.join("; ");
	}
	return JSON.stringify(detail);
}

export async function predictVideo(file: Blob, filename: string): Promise<PredictResponse> {
	const base = getInferenceApiBase();
	const form = new FormData();
	form.append("file", file, filename);

	const res = await fetch(`${base}/predict`, { method: "POST", body: form });

	if (!res.ok) {
		let message = res.statusText || `HTTP ${res.status}`;
		try {
			const j = (await res.json()) as { detail?: unknown };
			if (j?.detail != null) message = normalizeDetail(j.detail);
		} catch {
			/* ignore */
		}
		throw new Error(message);
	}

	return res.json() as Promise<PredictResponse>;
}

export async function checkInferenceHealth(): Promise<{ status: string; models_loaded: boolean }> {
	const base = getInferenceApiBase();
	const res = await fetch(`${base}/health`);
	if (!res.ok) throw new Error("Inference health check failed");
	return res.json() as Promise<{ status: string; models_loaded: boolean }>;
}
