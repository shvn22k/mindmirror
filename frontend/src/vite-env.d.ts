/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_NAME: string
  readonly VITE_PUBLIC_URL: string
  readonly VITE_API_URL: string
  readonly VITE_ENABLE_SOURCE_MAPPING: string
  readonly VITE_ENABLE_SSR: string
  /** Full origin of FastAPI inference API; omit in dev to use same-origin `/inference` (Vite proxy). */
  readonly VITE_INFERENCE_API_URL?: string
  /** Clerk Dashboard → API Keys → Publishable key */
  readonly VITE_CLERK_PUBLISHABLE_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
