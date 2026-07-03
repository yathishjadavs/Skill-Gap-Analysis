// Axios API wrapper. All backend responses use the { success, data, message,
// errors } envelope. Helpers unwrap `data` on success and throw a normalized
// Error (with `.status` and `.errors`) on failure.
import axios from "axios";
 
const BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000/api/v1";
 
const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});
 
function normalizeError(error) {
  const resp = error.response;
  const payload = resp && resp.data ? resp.data : {};
  const message =
    payload.message ||
    (error.code === "ECONNABORTED"
      ? "Request timed out. Analysis can take a few minutes with slower models."
      : error.message) ||
    "Unexpected error";
  const err = new Error(message);
  err.status = resp ? resp.status : 0;
  err.errors = payload.errors || null;
  return err;
}
 
async function unwrap(promise) {
  try {
    const res = await promise;
    return res.data && "data" in res.data ? res.data.data : res.data;
  } catch (error) {
    throw normalizeError(error);
  }
}
 
// ---- Health ----
export const getHealth = () => unwrap(client.get("/health"));
 
// ---- Admin ----
export const getCategories = () =>
  unwrap(client.get("/admin/categories")).then((d) => d.categories || []);
 
export const listRoles = () =>
  unwrap(client.get("/admin/roles")).then((d) => d.roles || []);
 
export const getRole = (slug) =>
  unwrap(client.get(`/admin/roles/${slug}`)).then((d) => d.role);
 
export const createRole = (payload) =>
  unwrap(client.post("/admin/roles", payload)).then((d) => d.role);
 
export const updateRole = (slug, payload) =>
  unwrap(client.put(`/admin/roles/${slug}`, payload)).then((d) => d.role);
 
export const deleteRole = (slug) => unwrap(client.delete(`/admin/roles/${slug}`));
 
// ---- Employee ----
export const listEmployeeRoles = () =>
  unwrap(client.get("/employee/roles")).then((d) => d.roles || []);
 
export const analyzeResume = ({ file, roleSlug, employeeName }) => {
  const form = new FormData();
  form.append("resume", file);
  form.append("role_slug", roleSlug);
  if (employeeName) form.append("employee_name", employeeName);
  return unwrap(
    client.post("/employee/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
      // Two LLM calls; thinking models (gemini-2.5-*) can be slow, so allow ~4 min.
      timeout: 240000,
    })
  );
};
 
export default client;
 
 