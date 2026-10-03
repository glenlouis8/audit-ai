"use client";

import { useEffect } from "react";

export default function WarmUp() {
  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/health`).catch(() => {});
  }, []);

  return null;
}
