# MIRA Interface

A clean, minimalist chat interface with sophisticated animations and theme support.

## Files

- `mira-interface-simplified.html` - Core HTML and CSS
- `functional.js` - Core functionality, state management, and data operations
- `aesthetic.js` - Animations and visual effects
- `mira-demo-data.js` - Optional demo data for testing/presentation

## Architecture

The JavaScript is split into two logical modules:

### functional.js
- State management (theme, navigation state)
- Event handlers (clicks, keyboard, search)
- Data operations (conversation filtering, calendar)
- Core UI logic (toggles, message handling)
- Initialization and event binding

### aesthetic.js
- Pinball plunger animation system
- ASCII loading screen animations
- Response transition effects
- All visual animations and timing

## Usage

### Production Mode
Remove the demo data script tag from the HTML. The interface will work with:
- Empty conversation history
- Default responses
- No ghost text animations
- No simulated activity badges

### Demo Mode
Include all files to enable:
- Pre-populated conversation history
- Rotating tip responses
- Ghost text suggestions
- Simulated tool and workflow activity

## Features

- **Dark/Light Theme Toggle** - Automatic detection of system preference
- **Conversation History** - Searchable, filterable by date
- **Calendar Integration** - Visual date picker
- **Smooth Animations** - Loading screen, message transitions, pinball plunger effect
- **Workflow Visualization** - Step-by-step progress indicators
- **Responsive Design** - Works on desktop and mobile

## Customization

To integrate with a real backend:
1. Remove the demo data script tag
2. Replace `renderConversations()` with API calls
3. Update `showResponse()` to fetch from your server
4. Remove demo activity simulation calls

The interface is designed to gracefully degrade when demo functions aren't available.