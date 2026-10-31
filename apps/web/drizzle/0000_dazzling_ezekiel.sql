CREATE TABLE "changes" (
	"id" text PRIMARY KEY NOT NULL,
	"env" text NOT NULL,
	"title" text NOT NULL,
	"card" jsonb NOT NULL,
	"pr_url" text,
	"created_at" timestamp with time zone NOT NULL,
	CONSTRAINT "changes_id_format" CHECK ("changes"."id" ~ '^c[0-9]{3,}$' and "changes"."id" !~ '^c0+$'),
	CONSTRAINT "changes_card_object" CHECK (jsonb_typeof("changes"."card") = 'object')
);
--> statement-breakpoint
CREATE TABLE "evaluations" (
	"id" uuid PRIMARY KEY NOT NULL,
	"change_id" text NOT NULL,
	"verdict" text NOT NULL,
	"alpha" double precision NOT NULL,
	"lambda" double precision NOT NULL,
	"evals_run" integer NOT NULL,
	"evals_total" integer NOT NULL,
	"hours_saved_pct" double precision,
	"model_version" text NOT NULL,
	"created_at" timestamp with time zone NOT NULL,
	CONSTRAINT "evaluations_verdict" CHECK ("evaluations"."verdict" in ('pass', 'block')),
	CONSTRAINT "evaluations_alpha" CHECK ("evaluations"."alpha" between 0 and 1),
	CONSTRAINT "evaluations_lambda" CHECK ("evaluations"."lambda" between 0 and 1),
	CONSTRAINT "evaluations_counts" CHECK ("evaluations"."evals_run" >= 0 and "evaluations"."evals_total" >= "evaluations"."evals_run"),
	CONSTRAINT "evaluations_hours" CHECK ("evaluations"."hours_saved_pct" between 0 and 100)
);
--> statement-breakpoint
CREATE TABLE "failures" (
	"id" uuid PRIMARY KEY NOT NULL,
	"pair_id" uuid NOT NULL,
	"keyframe_urls" jsonb NOT NULL,
	"overlay_urls" jsonb NOT NULL,
	"explanation" jsonb NOT NULL,
	CONSTRAINT "failures_keyframes_array" CHECK (jsonb_typeof("failures"."keyframe_urls") = 'array'),
	CONSTRAINT "failures_overlays_array" CHECK (jsonb_typeof("failures"."overlay_urls") = 'array'),
	CONSTRAINT "failures_explanation_object" CHECK (jsonb_typeof("failures"."explanation") = 'object')
);
--> statement-breakpoint
CREATE TABLE "pairs" (
	"id" uuid PRIMARY KEY NOT NULL,
	"evaluation_id" uuid NOT NULL,
	"scenario_id" text NOT NULL,
	"p_regress" double precision NOT NULL,
	"action" text NOT NULL,
	"result" text,
	"success_before" double precision,
	"success_after" double precision,
	CONSTRAINT "pairs_probability" CHECK ("pairs"."p_regress" between 0 and 1),
	CONSTRAINT "pairs_success_before" CHECK ("pairs"."success_before" is null or "pairs"."success_before" between 0 and 1),
	CONSTRAINT "pairs_success_after" CHECK ("pairs"."success_after" is null or "pairs"."success_after" between 0 and 1)
);
--> statement-breakpoint
CREATE TABLE "reports" (
	"id" uuid PRIMARY KEY NOT NULL,
	"suite" text NOT NULL,
	"created_at" timestamp with time zone NOT NULL,
	"body" jsonb NOT NULL,
	CONSTRAINT "reports_suite" CHECK ("reports"."suite" in ('robot', 'software')),
	CONSTRAINT "reports_body_object" CHECK (jsonb_typeof("reports"."body") = 'object')
);
--> statement-breakpoint
ALTER TABLE "evaluations" ADD CONSTRAINT "evaluations_change_id_changes_id_fk" FOREIGN KEY ("change_id") REFERENCES "public"."changes"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "failures" ADD CONSTRAINT "failures_pair_id_pairs_id_fk" FOREIGN KEY ("pair_id") REFERENCES "public"."pairs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "pairs" ADD CONSTRAINT "pairs_evaluation_id_evaluations_id_fk" FOREIGN KEY ("evaluation_id") REFERENCES "public"."evaluations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE UNIQUE INDEX "pairs_evaluation_scenario_unique" ON "pairs" USING btree ("evaluation_id","scenario_id");
