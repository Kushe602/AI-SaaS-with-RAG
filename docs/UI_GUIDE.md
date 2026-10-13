# DocuChat UI guide

## Design direction

The templates use a modern dark workspace style with a deep navy page background, restrained violet accents, subtle borders, and high-contrast text. The styling is implemented in `app/templates/base.html` so shared pages inherit the same visual system without changing the route or form contracts.

## Main screens

- **Landing page** (`app/templates/landing.html`): explains document Q&A, retrieval, citations, and the first steps; sign-in and registration links remain the existing routes.
- **Dashboard** (`app/templates/dashboard.html`): document management, conversation creation, optional document scoping, plan and usage information.
- **Chat** (`app/templates/chat.html`): conversation UI, streamed responses, and source citations.
- **Authentication** (`app/templates/login.html`, `register.html`): existing account workflows styled by the shared base template.

## Interaction and accessibility

- Existing form actions and HTMX endpoints are retained.
- Visible focus indicators are provided for keyboard users.
- Reduced-motion preferences are respected by shared styles.
- The landing-page chat preview is illustrative only; it does not make a network request or pretend to be a working chat.

## Verify locally

1. Follow the root README quickstart and start the app.
2. Check landing, registration, login, dashboard, document upload, conversation creation, streaming answers, and citations.
3. Check the layout at narrow and wide viewport widths.
4. Run the existing test suite using the command documented in the README.

## Scope

This guide describes the interface and intended interactions. It does not claim that the application has been executed or browser-tested as part of this UI update.
