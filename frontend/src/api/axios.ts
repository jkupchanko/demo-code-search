import axios from "axios";

/**
 * The API is served from the same origin as the app: the functions in /api and
 * this bundle are one Vercel deployment. There is no backend URL to configure,
 * and no CORS, because there is no second origin.
 *
 * In development, Vite proxies /api to `vercel dev` (see vite.config.ts).
 */
export const api = axios.create({
  baseURL: "/",
  validateStatus: () => true,
});
