/**
 * Collapse Animation
 *
 * Converts text to the alternate font character-by-character, then drops the
 * characters with gravity. Character widths are measured and locked before the
 * font swap so nothing reflows. Used by core.js Functions.collapseSegment.
 *
 * Required CSS classes:
 *   .converted { font-family: 'YourFont', monospace; }
 *   .falling { pointer-events: none; }
 *
 * Usage:
 *   collapseAnimation(element, {
 *     scrollContainer: element,   // overflow hidden during the fall
 *     onFallComplete: () => {}
 *   });
 */

const COLLAPSE_CONVERSION_DURATION_MS = 5000;  // Total time to convert all chars
const COLLAPSE_FALL_DURATION_MS = 5000;        // Total time to stagger all falls
const COLLAPSE_MAX_MS_PER_CHAR = 30;           // Cap on per-character timing
const COLLAPSE_GRAVITY = 980 * 1.9;            // Gravity acceleration (px/s²)

/**
 * Fisher-Yates shuffle
 */
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

/**
 * Converts the element's text, then drops it with gravity.
 *
 * @param {HTMLElement} element - Element containing text to animate
 * @param {Object} options
 * @param {HTMLElement} options.scrollContainer - Element whose overflow is hidden during the fall (prevents scrollbar flash)
 * @param {Function} options.onFallComplete - Called when every character has fallen off screen
 */
function collapseAnimation(element, { scrollContainer, onFallComplete }) {
    // Step 1: Wrap non-space characters in spans while preserving DOM structure
    const spans = [];

    function wrapTextNode(textNode) {
        const text = textNode.textContent;
        const fragment = document.createDocumentFragment();

        for (const char of text) {
            if (char === ' ' || char === '\n' || char === '\t') {
                fragment.appendChild(document.createTextNode(char));
            } else {
                const span = document.createElement('span');
                span.textContent = char;
                fragment.appendChild(span);
                spans.push(span);
            }
        }

        textNode.parentNode.replaceChild(fragment, textNode);
    }

    // Recursively walk all text nodes in the tree
    function walkAndWrap(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            wrapTextNode(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            // Process children (use array copy since we're modifying the tree)
            Array.from(node.childNodes).forEach(walkAndWrap);
        }
    }

    walkAndWrap(element);

    // Step 2: Measure widths (forces layout)
    const widths = spans.map(s => s.getBoundingClientRect().width);

    // Step 3: Lock widths to prevent reflow during font change
    spans.forEach((span, i) => {
        span.style.display = 'inline-block';
        span.style.width = widths[i] + 'px';
        span.style.textAlign = 'center';
        span.style.position = 'relative';
        span.style.verticalAlign = 'text-top';
    });

    // Step 4: Convert all characters, then fall.
    // Time-based: each tick batch-converts all chars due by elapsed time,
    // so hidden-tab throttling (1Hz) doesn't slow the conversion phase.
    const msPerChar = Math.min(COLLAPSE_MAX_MS_PER_CHAR, COLLAPSE_CONVERSION_DURATION_MS / spans.length);
    let index = 0;
    const conversionStart = performance.now();

    const interval = setInterval(() => {
        const target = Math.min(
            Math.floor((performance.now() - conversionStart) / msPerChar),
            spans.length
        );
        while (index < target) {
            spans[index].classList.add('converted');
            index++;
        }
        if (index >= spans.length) {
            clearInterval(interval);
            startGravityFall(spans, scrollContainer, onFallComplete);
        }
    }, msPerChar);
}

/**
 * Starts the gravity fall phase
 */
function startGravityFall(spans, scrollContainer, onFallComplete) {
    // Hide scrollbar during fall to prevent scrollbar flash as chars drop
    const originalOverflow = scrollContainer.style.overflow;
    scrollContainer.style.overflow = 'hidden';

    const staggerMs = Math.min(COLLAPSE_MAX_MS_PER_CHAR, COLLAPSE_FALL_DURATION_MS / spans.length);
    const shuffled = shuffle([...spans]);

    const items = shuffled.map((span, i) => ({
        el: span,
        delay: i * staggerMs,
        y: 0,
        vy: 0,
        // Random gravity multiplier: 0.85 to 1.15 (±15%)
        gravity: COLLAPSE_GRAVITY * (0.85 + Math.random() * 0.3),
        active: false
    }));

    let startTime = 0;
    let lastTime = 0;

    function tick(now) {
        if (!startTime) startTime = now;
        if (!lastTime) lastTime = now;

        const elapsed = now - startTime;
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        for (let i = items.length - 1; i >= 0; i--) {
            const item = items[i];
            if (elapsed < item.delay) continue;

            if (!item.active) {
                item.active = true;
                item.el.classList.add('falling');
            }

            item.vy += item.gravity * dt;
            item.y += item.vy * dt;
            item.el.style.transform = `translateY(${item.y}px)`;

            if (item.y > window.innerHeight) {
                item.el.style.visibility = 'hidden';
                items.splice(i, 1);
            }
        }

        if (items.length > 0) {
            requestAnimationFrame(tick);
        } else {
            scrollContainer.style.overflow = originalOverflow;
            onFallComplete();
        }
    }

    requestAnimationFrame(tick);
}

window.collapseAnimation = collapseAnimation;
