# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Start development server (localhost:3000)
npm run build    # Production build
npm run start    # Run production server
npm run lint     # ESLint
```

No test suite is configured.

## Environment

- `NEXT_PUBLIC_API_URL` — Backend base URL (default: `http://localhost:8000`)

## Architecture

**Next.js 16 App Router** frontend for an AI-powered NIST compliance auditing tool.

**Page structure:**
- `/` (`src/app/page.tsx`) — Landing page with hero, architecture overview, and "Launch Console" CTA
- `/chat` (`src/app/chat/page.tsx`) — Wraps `ChatInterface` full-screen

**Core component:** `src/components/ChatInterface.tsx`
- Sends `POST ${NEXT_PUBLIC_API_URL}/chat` with `{ query, history }` to an external FastAPI backend
- Handles **Server-Sent Events (SSE)** via `ReadableStream` + `TextDecoder` for streaming responses
- Parses newline-delimited JSON: `{ type: "token", content }` builds the message incrementally; `{ type: "sources", sources }` populates citation cards
- Citations link into a NIST CSF 2.0 PDF with page anchors
- Uses `react-markdown` for rendering assistant messages
- Includes a server cold-start warning (free tier notice: 45–60s)

**No API routes** — all backend calls go to the external `NEXT_PUBLIC_API_URL`.

**Styling:** Tailwind CSS v4 (dark emerald theme) + Framer Motion animations + Lucide icons.
