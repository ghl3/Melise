// Server-side auth for the worker upstream. Two independent layers:
//
//  - GCP_SA_KEY (service-account JSON, Vercel env only): mint a Google
//    ID token per request so an IAM-gated Cloud Run worker
//    (--no-allow-unauthenticated) admits us. Unset in local dev, where
//    the worker is a plain localhost process.
//  - MODEL_SERVER_TOKEN: the app-level shared secret. Sent as
//    X-Serve-Token when the Authorization header is occupied by the ID
//    token, else as a plain bearer.
//
// Never import this from client components — it holds credentials.

import { GoogleAuth, IdTokenClient } from "google-auth-library";

export const WORKER = process.env.MODEL_SERVER_URL ?? "http://localhost:8000";

let idClient: IdTokenClient | null = null;

async function idToken(): Promise<string | null> {
  const key = process.env.GCP_SA_KEY;
  if (!key) return null;
  if (!idClient) {
    const auth = new GoogleAuth({ credentials: JSON.parse(key) });
    // Audience must be the service's run.app origin, not a path.
    idClient = await auth.getIdTokenClient(new URL(WORKER).origin);
  }
  return idClient.idTokenProvider.fetchIdToken(new URL(WORKER).origin);
}

export async function upstreamHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  const token = process.env.MODEL_SERVER_TOKEN;
  const gcp = await idToken();
  if (gcp) {
    headers["Authorization"] = `Bearer ${gcp}`;
    if (token) headers["X-Serve-Token"] = token;
  } else if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}
