// Aesthetic JavaScript for MIRA Interface - Animations and Visual Effects

const asciiChars = ['.', '+', '*', 'o', "'", '-', '~', '|'];

// Animations (preserved as requested)
async function runPlungerAnimation(messageText) {
    const totalDuration = 1700;
    const pullbackDistance = 7;
    const launchDistance = -2;
    
    const animationPromise = animatePlunger(pullbackDistance, launchDistance, totalDuration);
    const fadePromise = animateTextDissolve(totalDuration);
    const projectilePromise = animateTextProjectile(messageText, totalDuration);
    
    await Promise.all([animationPromise, fadePromise, projectilePromise]);
}

function animateTextDissolve(totalDuration) {
    return new Promise((resolve) => {
        const startTime = performance.now();
        const duration = totalDuration * 0.10;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const alpha = 1 - progress;
            const isLightTheme = theme === 'light';
            if (isLightTheme) {
                elements.messageInput.style.color = `rgba(0, 0, 0, ${alpha})`;
            } else {
                elements.messageInput.style.color = `rgba(255, 255, 255, ${alpha})`;
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                elements.messageInput.value = '';
                elements.messageInput.style.color = '';
                elements.messageInput.focus();
                resolve();
            }
        };
        
        requestAnimationFrame(animate);
    });
}

function animateTextProjectile(messageText, totalDuration) {
    return new Promise((resolve) => {
        const projectile = document.createElement('div');
        projectile.className = 'message-projectile';
        projectile.textContent = messageText;
        
        elements.inputContainer.insertBefore(projectile, elements.inputContainer.firstChild);
        
        const strokeDistance = 150;
        const startTime = performance.now();
        const duration = totalDuration * 0.5;
        const launchStart = totalDuration * 0.18;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            let distance = 0;
            let opacity = 0;
            
            if (elapsed < launchStart) {
                distance = 0;
                opacity = 0;
            } else {
                const activeElapsed = elapsed - launchStart;
                const activeDuration = duration - launchStart;
                const activeProgress = Math.min(activeElapsed / activeDuration, 1);
                
                const eased = 1 - Math.pow(1 - activeProgress, 2.5);
                distance = eased * strokeDistance;
                opacity = 1 - activeProgress;
            }
            
            projectile.style.transform = `translateY(-${distance}px)`;
            projectile.style.opacity = opacity;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                if (projectile.parentNode) {
                    projectile.parentNode.removeChild(projectile);
                }
                resolve();
            }
        };
        
        requestAnimationFrame(animate);
    });
}

function animatePlunger(pullbackDistance, launchDistance, totalDuration) {
    return new Promise((resolve) => {
        const startTime = performance.now();
        
        const pullbackEnd = 0.12;
        const launchEnd = 0.40;
        const returnEnd = 0.55;
        const overshootEnd = 0.60;
        const settleEnd = 0.65;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / totalDuration, 1);
            
            let position = 0;
            
            if (progress <= pullbackEnd) {
                const phaseProgress = progress / pullbackEnd;
                const eased = 1 - Math.pow(1 - phaseProgress, 2);
                position = eased * pullbackDistance;
                
            } else if (progress <= launchEnd) {
                const phaseProgress = (progress - pullbackEnd) / (launchEnd - pullbackEnd);
                const eased = 1 - Math.exp(-8 * phaseProgress);
                position = pullbackDistance + eased * (launchDistance - pullbackDistance);
                
            } else if (progress <= returnEnd) {
                const phaseProgress = (progress - launchEnd) / (returnEnd - launchEnd);
                const eased = phaseProgress * phaseProgress;
                position = launchDistance + eased * (0 - launchDistance);
                
            } else if (progress <= overshootEnd) {
                const phaseProgress = (progress - returnEnd) / (overshootEnd - returnEnd);
                const eased = Math.sin(phaseProgress * Math.PI / 2);
                position = eased * 1;
                
            } else if (progress <= settleEnd) {
                const phaseProgress = (progress - overshootEnd) / (settleEnd - overshootEnd);
                const eased = 1 - Math.pow(1 - phaseProgress, 2);
                position = 1 - eased;
            } else {
                position = 0;
            }
            
            elements.inputContainer.style.transform = `translateY(${position}px)`;
            
            if (progress < settleEnd) {
                requestAnimationFrame(animate);
            } else {
                elements.inputContainer.style.transform = '';
                resolve();
            }
        };
        
        requestAnimationFrame(animate);
    });
}


// Loading animation
function showLoadingScreen(callback) {
    const chars = prepareAscii();
    elements.loadingScreen.classList.add('active');
    
    animateAscii(chars);
    
    setTimeout(() => {
        despawnAscii(chars);
    }, 4300);
    
    setTimeout(callback, 4700);
    
    setTimeout(() => {
        elements.loadingScreen.classList.remove('active');
    }, 5400);
}

function prepareAscii() {
    elements.asciiContainer.innerHTML = '';
    const lineLength = Math.floor(window.innerWidth / 10);
    const chars = [];
    
    for (let i = 0; i < lineLength; i++) {
        const span = document.createElement('span');
        span.className = 'ascii-char';
        
        // Always place characters at first and last positions
        if (i === 0 || i === lineLength - 1) {
            span.textContent = asciiChars[Math.floor(Math.random() * asciiChars.length)];
        } else if (Math.random() < 0.2) {
            span.textContent = asciiChars[Math.floor(Math.random() * asciiChars.length)];
        } else {
            span.textContent = '\u00A0';
        }
        
        elements.asciiContainer.appendChild(span);
        chars.push(span);
    }
    
    return chars;
}

function animateAscii(chars) {
    const visibleChars = chars.filter(c => c.textContent.trim());
    const interval = 550 / visibleChars.length;
    
    visibleChars.forEach((char, i) => {
        setTimeout(() => char.classList.add('visible'), i * interval);
    });
    
    setTimeout(() => {
        const randomInterval = setInterval(() => {
            visibleChars.forEach(char => {
                if (char.classList.contains('visible')) {
                    char.textContent = asciiChars[Math.floor(Math.random() * asciiChars.length)];
                }
            });
        }, 173);
        
        setTimeout(() => clearInterval(randomInterval), 3750);
    }, 550);
}

function despawnAscii(chars) {
    const center = Math.floor(chars.length / 2);
    const interval = 400 / (center + 1);
    
    for (let i = 0; i <= center; i++) {
        setTimeout(() => {
            if (center - i >= 0) chars[center - i].classList.remove('visible');
            if (center + i < chars.length) chars[center + i].classList.remove('visible');
        }, i * interval);
    }
}

// Badge activation system
function activateBadges() {
    const toolBadge = elements.toolBadge;
    const workflowBadge = elements.workflowBadge;
    
    console.log('Activating badges...', { toolBadge, workflowBadge });
    
    if (toolBadge) {
        // Add active class
        toolBadge.classList.add('active');
        console.log('Tool badge classes after adding active:', toolBadge.className);
        console.log('Tool badge computed box-shadow:', getComputedStyle(toolBadge).boxShadow);
        
        // Remove active class after flicker animation completes
        setTimeout(() => {
            toolBadge.classList.remove('active');
            console.log('Removed active from tool badge');
        }, 3000);
    }
    
    if (workflowBadge) {
        // Add active class
        workflowBadge.classList.add('active');
        console.log('Workflow badge classes after adding active:', workflowBadge.className);
        
        // Keep active class (workflow badge stays lit)
        // Note: No timeout to remove 'active' class for workflow badge
    }
}

