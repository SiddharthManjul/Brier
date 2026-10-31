import { sql } from "drizzle-orm";
import {
  check, doublePrecision, integer, jsonb, pgTable, text, timestamp, uniqueIndex, uuid,
} from "drizzle-orm/pg-core";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export const changes = pgTable("changes", {
  id: text("id").primaryKey(),
  env: text("env").notNull(),
  title: text("title").notNull(),
  card: jsonb("card").$type<Record<string, Json>>().notNull(),
  prUrl: text("pr_url"),
  createdAt: timestamp("created_at", { withTimezone: true, mode: "string" }).notNull(),
}, (table) => [
  check("changes_id_format", sql`${table.id} ~ '^c[0-9]{3,}$' and ${table.id} !~ '^c0+$'`),
  check("changes_card_object", sql`jsonb_typeof(${table.card}) = 'object'`),
]);

export const evaluations = pgTable("evaluations", {
  id: uuid("id").primaryKey(),
  changeId: text("change_id").notNull().references(() => changes.id),
  verdict: text("verdict").$type<"pass" | "block">().notNull(),
  alpha: doublePrecision("alpha").notNull(),
  lambda: doublePrecision("lambda").notNull(),
  evalsRun: integer("evals_run").notNull(),
  evalsTotal: integer("evals_total").notNull(),
  hoursSavedPct: doublePrecision("hours_saved_pct"),
  modelVersion: text("model_version").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true, mode: "string" }).notNull(),
}, (table) => [
  check("evaluations_verdict", sql`${table.verdict} in ('pass', 'block')`),
  check("evaluations_alpha", sql`${table.alpha} between 0 and 1`),
  check("evaluations_lambda", sql`${table.lambda} between 0 and 1`),
  check("evaluations_counts", sql`${table.evalsRun} >= 0 and ${table.evalsTotal} >= ${table.evalsRun}`),
  check("evaluations_hours", sql`${table.hoursSavedPct} between 0 and 100`),
]);

export const pairs = pgTable("pairs", {
  id: uuid("id").primaryKey(),
  evaluationId: uuid("evaluation_id").notNull()
    .references(() => evaluations.id, { onDelete: "cascade" }),
  scenarioId: text("scenario_id").notNull(),
  pRegress: doublePrecision("p_regress").notNull(),
  action: text("action").notNull(),
  result: text("result"),
  successBefore: doublePrecision("success_before"),
  successAfter: doublePrecision("success_after"),
}, (table) => [
  uniqueIndex("pairs_evaluation_scenario_unique").on(table.evaluationId, table.scenarioId),
  check("pairs_probability", sql`${table.pRegress} between 0 and 1`),
  check("pairs_success_before", sql`${table.successBefore} is null or ${table.successBefore} between 0 and 1`),
  check("pairs_success_after", sql`${table.successAfter} is null or ${table.successAfter} between 0 and 1`),
]);

export const failures = pgTable("failures", {
  id: uuid("id").primaryKey(),
  pairId: uuid("pair_id").notNull().references(() => pairs.id, { onDelete: "cascade" }),
  keyframeUrls: jsonb("keyframe_urls").$type<string[]>().notNull(),
  overlayUrls: jsonb("overlay_urls").$type<string[]>().notNull(),
  explanation: jsonb("explanation").$type<Record<string, Json>>().notNull(),
}, (table) => [
  check("failures_keyframes_array", sql`jsonb_typeof(${table.keyframeUrls}) = 'array'`),
  check("failures_overlays_array", sql`jsonb_typeof(${table.overlayUrls}) = 'array'`),
  check("failures_explanation_object", sql`jsonb_typeof(${table.explanation}) = 'object'`),
]);

export const reports = pgTable("reports", {
  id: uuid("id").primaryKey(),
  suite: text("suite").$type<"robot" | "software">().notNull(),
  createdAt: timestamp("created_at", { withTimezone: true, mode: "string" }).notNull(),
  body: jsonb("body").$type<Record<string, Json>>().notNull(),
}, (table) => [
  check("reports_suite", sql`${table.suite} in ('robot', 'software')`),
  check("reports_body_object", sql`jsonb_typeof(${table.body}) = 'object'`),
]);
