import React, { useEffect, useState, useRef } from "react";
import { Play, AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import RoleSelector from "../components/employee/RoleSelector";
import ResumeUpload from "../components/employee/ResumeUpload";
import AnalysisResults from "../components/employee/AnalysisResults";
import LoadingSpinner from "../components/shared/LoadingSpinner";
import { listEmployeeRoles, analyzeResume } from "../services/api";
 
const PIPELINE_STEPS = [
  "Parsing resume",
  "Chunking & embedding",
  "Indexing & semantic retrieval",
  "RAG context assembly",
  "LLM skill extraction",
  "Gap comparison",
  "Course retrieval & re-ranking",
];
 
// Rotating, human-friendly status lines shown while the pipeline runs.
const STATUS_MESSAGES = [
  "Reading your resume",
  "Extracting your experience and projects",
  "Understanding the skills you've demonstrated",
  "Matching your profile against the target role",
  "Estimating your proficiency levels",
  "Identifying skill gaps",
  "Searching thousands of courses",
  "Curating a personalized learning path",
  "Finalizing your readiness report",
];
 
export default function EmployeePage() {
  const [roles, setRoles] = useState([]);
  const [rolesLoading, setRolesLoading] = useState(true);
  const [rolesError, setRolesError] = useState(null);
 
  const [roleSlug, setRoleSlug] = useState("");
  const [employeeName, setEmployeeName] = useState("");
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState(null);
 
  const [phase, setPhase] = useState("form"); // form | analyzing | results
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
 
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const data = await listEmployeeRoles();
        if (active) setRoles(data);
      } catch (e) {
        if (active) setRolesError(e.message);
      } finally {
        if (active) setRolesLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);
 
  const canSubmit = roleSlug && file && !fileError;
 
  const handleAnalyze = async () => {
    if (!canSubmit) return;
    setError(null);
    setPhase("analyzing");
    try {
      const data = await analyzeResume({ file, roleSlug, employeeName });
      setResult(data);
      setPhase("results");
    } catch (e) {
      setError(
        e.errors && e.errors.length ? `${e.message}: ${e.errors.join("; ")}` : e.message
      );
      setPhase("form");
    }
  };
 
  const reset = () => {
    setResult(null);
    setFile(null);
    setPhase("form");
    setError(null);
  };
 
  if (phase === "results" && result) {
    return (
      <div className="page">
        <AnalysisResults result={result} onReset={reset} />
      </div>
    );
  }
 
  if (phase === "analyzing") {
    return (
      <div className="page">
        <AnalyzingView />
      </div>
    );
  }
 
  return (
    <div className="page">
      <div className="page-header">
        <h1>Skill Gap Analysis</h1>
        <div className="subtitle">
          Select a target role and upload your resume to assess role readiness.
        </div>
      </div>
 
      {rolesError ? (
        <div className="alert alert-danger mb-4">
          <AlertTriangle size={18} />
          <div>{rolesError}</div>
        </div>
      ) : null}
 
      {error ? (
        <div className="alert alert-danger mb-4">
          <AlertTriangle size={18} />
          <div>{error}</div>
        </div>
      ) : null}
 
      <div className="card card-body" style={{ maxWidth: 680 }}>
        {rolesLoading ? (
          <LoadingSpinner label="Loading roles..." />
        ) : roles.length === 0 ? (
          <div className="alert alert-info">
            <AlertTriangle size={18} />
            <div>
              No roles are defined yet. Ask an administrator to create a role first.
            </div>
          </div>
        ) : (
          <>
            <div className="field">
              <RoleSelector roles={roles} value={roleSlug} onChange={setRoleSlug} />
            </div>
 
            <div className="field">
              <label className="label">Your Name</label>
              <input
                className="input"
                placeholder="Optional"
                value={employeeName}
                onChange={(e) => setEmployeeName(e.target.value)}
              />
            </div>
 
            <ResumeUpload
              file={file}
              onFileSelect={setFile}
              onClear={() => setFile(null)}
              onError={setFileError}
            />
 
            {fileError ? (
              <div className="alert alert-danger mt-4">
                <AlertTriangle size={18} />
                <div>{fileError}</div>
              </div>
            ) : null}
 
            <button
              className="btn btn-primary btn-block mt-6"
              disabled={!canSubmit}
              onClick={handleAnalyze}
            >
              <Play size={16} /> Run analysis
            </button>
          </>
        )}
      </div>
    </div>
  );
}
 
function AnalyzingView() {
  const [activeStep, setActiveStep] = useState(0);
  const [msgIndex, setMsgIndex] = useState(0);
  const [dots, setDots] = useState("");
  const stepTimer = useRef(null);
  const msgTimer = useRef(null);
  const dotTimer = useRef(null);
 
  useEffect(() => {
    // Cosmetic progression through pipeline stages while the request runs.
    stepTimer.current = window.setInterval(() => {
      setActiveStep((s) => Math.min(s + 1, PIPELINE_STEPS.length - 1));
    }, 3000);
    // Rotating human-friendly status message.
    msgTimer.current = window.setInterval(() => {
      setMsgIndex((m) => (m + 1) % STATUS_MESSAGES.length);
    }, 2400);
    // Animated trailing dots (Claude-style "working" indicator).
    dotTimer.current = window.setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 400);
    return () => {
      window.clearInterval(stepTimer.current);
      window.clearInterval(msgTimer.current);
      window.clearInterval(dotTimer.current);
    };
  }, []);
 
  return (
    <div className="card card-body" style={{ maxWidth: 620, margin: "40px auto" }}>
      <div className="center-col mb-4">
        <span className="spinner lg" />
        <div className="analyze-status" aria-live="polite">
          <span className="analyze-status-text">{STATUS_MESSAGES[msgIndex]}</span>
          <span className="analyze-dots">{dots}</span>
        </div>
        <div className="text-sm text-muted">
          Working through the analysis pipeline. This can take a few minutes.
        </div>
      </div>
 
      <div className="analyze-progress mb-4">
        <div className="analyze-progress-bar" />
      </div>
 
      <div className="steps">
        {PIPELINE_STEPS.map((label, i) => {
          const done = i < activeStep;
          const active = i === activeStep;
          return (
            <div
              key={label}
              className={"step" + (done ? " done" : active ? " active" : "")}
            >
              <span className="step-dot">
                {done ? (
                  <CheckCircle2 size={15} />
                ) : active ? (
                  <Loader2 size={14} className="spin-inline" />
                ) : (
                  i + 1
                )}
              </span>
              {label}
            </div>
          );
        })}
      </div>
    </div>
  );
}
 
 