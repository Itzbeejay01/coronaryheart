document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initNeuralBackground();
});

function initNavigation() {
    const navContainer = document.querySelector('[data-nav]');
    const toggleButton = document.querySelector('[data-nav-toggle]');

    if (!navContainer || !toggleButton) {
        return;
    }

    toggleButton.addEventListener('click', () => {
        navContainer.classList.toggle('is-open');
        const isOpen = navContainer.classList.contains('is-open');
        toggleButton.setAttribute('aria-expanded', String(isOpen));
    });

    navContainer.querySelectorAll('.nav-link, .nav-cta').forEach((link) => {
        link.addEventListener('click', () => {
            navContainer.classList.remove('is-open');
            toggleButton.setAttribute('aria-expanded', 'false');
        });
    });
}

function initNeuralBackground() {
    const container = document.getElementById('neuralBg');

    if (!container) {
        return;
    }

    const nodes = [];
    const connections = [];
    const numNodes = 24;

    const createNode = () => {
        const node = document.createElement('div');
        node.className = 'neural-node';
        node.style.left = `${Math.random() * 100}%`;
        node.style.top = `${Math.random() * 100}%`;
        node.style.animationDelay = `${Math.random() * 3}s`;
        container.appendChild(node);
        nodes.push(node);
    };

    for (let i = 0; i < numNodes; i += 1) {
        createNode();
    }

    for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
            if (Math.random() < 0.24) {
                const connection = document.createElement('div');
                connection.className = 'neural-connection';
                container.appendChild(connection);
                connections.push({ element: connection, start: nodes[i], end: nodes[j] });
            }
        }
    }

    const updateConnections = () => {
        const containerRect = container.getBoundingClientRect();

        connections.forEach((connection) => {
            const startRect = connection.start.getBoundingClientRect();
            const endRect = connection.end.getBoundingClientRect();

            const x1 = startRect.left + startRect.width / 2 - containerRect.left;
            const y1 = startRect.top + startRect.height / 2 - containerRect.top;
            const x2 = endRect.left + endRect.width / 2 - containerRect.left;
            const y2 = endRect.top + endRect.height / 2 - containerRect.top;

            const length = Math.hypot(x2 - x1, y2 - y1);
            const angle = Math.atan2(y2 - y1, x2 - x1);

            connection.element.style.width = `${length}px`;
            connection.element.style.left = `${x1}px`;
            connection.element.style.top = `${y1}px`;
            connection.element.style.transform = `rotate(${angle}rad)`;
        });
    };

    updateConnections();

    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(updateConnections, 120);
    });
}

