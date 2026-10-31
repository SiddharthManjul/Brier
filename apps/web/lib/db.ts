import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "../db/schema.ts";

/** Create the database client only when a request needs it. */
export function getDb() {
  const url = process.env.DATABASE_URL;
  if (!url?.trim()) throw new Error("DATABASE_URL is required");
  return drizzle(neon(url), { schema });
}
