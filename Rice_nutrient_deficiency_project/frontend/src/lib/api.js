// Prefer explicit env, else fall back to same-origin guess (useful when reverse-proxied)
const API_URL =
  import.meta.env.VITE_API_URL ||
  (window?.location?.origin?.includes(":5173")
    ? window.location.origin.replace(":5173", ":8000")
    : "http://localhost:8000");

export async function healthCheck() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function getTrainingAccuracies() {
  try {
    const res = await fetch(`${API_URL}/training-accuracies`);
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(txt || `Failed to fetch training accuracies (${res.status})`);
    }
    return await res.json();
  } catch (e) {
    console.error("getTrainingAccuracies error:", e);
    throw e;
  }
}

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file); // backend expects field name 'file'
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(errText || `Predict failed with ${res.status}`);
  }
  return res.json();
}

export async function predictAllModels(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/predict-all`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(errText || `Predict all failed with ${res.status}`);
  }
  return res.json();
}

export async function predictEnsemble(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/predict-ensemble`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(errText || `Ensemble predict failed with ${res.status}`);
  }
  return res.json();
}
