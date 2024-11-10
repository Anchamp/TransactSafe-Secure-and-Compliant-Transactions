document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts with animation
    initializeCharts();

    // Setup auto-refresh
    setupAutoRefresh();

    // Initialize table sorting and filtering
    initializeTableControls();
});

function initializeCharts() {
    // Set default Chart.js options
    Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.font.size = 13;
    Chart.defaults.color = '#4b5563';

    // Model Performance Chart
    createModelPerformanceChart();

    // Risk Distribution Chart
    createRiskDistributionChart();

    // Trend Chart
    createTrendChart();
}

function createModelPerformanceChart() {
    const ctx = document.getElementById('modelPerformanceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.modelPerformance.labels,
            datasets: [{
                label: 'Model Performance',
                data: chartData.modelPerformance.scores,
                backgroundColor: 'rgba(59, 130, 246, 0.8)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function createRiskDistributionChart() {
    const ctx = document.getElementById('riskDistributionChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartData.riskDistribution.labels,
            datasets: [{
                data: chartData.riskDistribution.values,
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',  // High Risk
                    'rgba(245, 158, 11, 0.8)', // Medium Risk
                    'rgba(34, 197, 94, 0.8)'   // Low Risk
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 2000,
                easing: 'easeOutQuart'
            },
            cutout: '60%'
        }
    });
}

function createTrendChart() {
    const ctx = document.getElementById('trendChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.trendData.dates,
            datasets: [{
                label: 'Daily Alerts',
                data: chartData.trendData.counts,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        stepSize: 1
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function setupAutoRefresh() {
    // Refresh dashboard data every 5 minutes
    const REFRESH_INTERVAL = 5 * 60 * 1000;

    setInterval(() => {
        fetch(window.location.href)
            .then(response => response.text())
            .then(html => {
                const parser = new DOMParser();
                const newDoc = parser.parseFromString(html, 'text/html');

                // Update summary cards
                updateSummaryCards(newDoc);

                // Update alerts table
                updateAlertsTable(newDoc);

                // Update charts if data changed
                if (hasDataChanged(newDoc)) {
                    initializeCharts();
                }
            })
            .catch(error => console.error('Error refreshing dashboard:', error));
    }, REFRESH_INTERVAL);
}

function initializeTableControls() {
    // Table sorting
    document.querySelectorAll('th').forEach(headerCell => {
        headerCell.addEventListener('click', () => {
            const tableBody = headerCell.closest('table').querySelector('tbody');
            const rows = Array.from(tableBody.querySelectorAll('tr'));
            const columnIndex = headerCell.cellIndex;

            const sortedRows = sortTableRows(rows, columnIndex);
            tableBody.innerHTML = '';
            tableBody.append(...sortedRows);
        });
    });

    // Risk filtering
    const riskFilter = document.getElementById('riskFilter');
    if (riskFilter) {
        riskFilter.addEventListener('change', (e) => {
            const selectedRisk = e.target.value;
            const rows = document.querySelectorAll('#alertsTable tbody trconst selectedRisk = e.target.value;
            const rows = document.querySelectorAll('#alertsTable tbody tr');

            rows.forEach(row => {
                if (selectedRisk === 'all') {
                    row.style.display = '';
                } else {
                    const badge = row.querySelector('.status-badge');
                    const rowRisk = badge ? badge.textContent.toLowerCase().replace(/\s+/g, '-') : '';
                    row.style.display = rowRisk === selectedRisk ? '' : 'none';
                }
            });
        });
    }
}

function sortTableRows(rows, columnIndex) {
    return rows.sort((a, b) => {
        const cellA = a.cells[columnIndex].textContent.trim();
        const cellB = b.cells[columnIndex].textContent.trim();

        // Handle currency values
        if (cellA.startsWith('$') && cellB.startsWith('$')) {
            const numA = parseFloat(cellA.replace(/[$,]/g, ''));
            const numB = parseFloat(cellB.replace(/[$,]/g, ''));
            return numA - numB;
        }

        // Handle risk scores
        if (!isNaN(cellA) && !isNaN(cellB)) {
            return parseFloat(cellA) - parseFloat(cellB);
        }

        // Handle dates
        const dateA = new Date(cellA);
        const dateB = new Date(cellB);
        if (!isNaN(dateA) && !isNaN(dateB)) {
            return dateA - dateB;
        }

        // Default string comparison
        return cellA.localeCompare(cellB);
    });
}

function updateSummaryCards(newDoc) {
    const summaryCards = document.querySelectorAll('.summary-value');
    const newSummaryCards = newDoc.querySelectorAll('.summary-value');

    summaryCards.forEach((card, index) => {
        if (newSummaryCards[index]) {
            const oldValue = parseFloat(card.textContent.replace(/[$,]/g, ''));
            const newValue = parseFloat(newSummaryCards[index].textContent.replace(/[$,]/g, ''));

            if (oldValue !== newValue) {
                animateNumberChange(card, oldValue, newValue);
            }
        }
    });
}

function animateNumberChange(element, start, end) {
    const duration = 1000;
    const startTime = performance.now();
    const isCurrency = element.textContent.includes('$');

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function for smooth animation
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeOutCubic;

        element.textContent = isCurrency
            ? `$${current.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
            : Math.round(current).toLocaleString('en-US');

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function updateAlertsTable(newDoc) {
    const currentTable = document.getElementById('alertsTable');
    const newTable = newDoc.getElementById('alertsTable');

    if (currentTable && newTable) {
        const currentBody = currentTable.querySelector('tbody');
        const newBody = newTable.querySelector('tbody');

        // Animate rows that are being updated
        const currentRows = Array.from(currentBody.children);
        const newRows = Array.from(newBody.children);

        // Remove old rows
        currentRows.forEach(row => {
            row.classList.add('fade-out');
            setTimeout(() => row.remove(), 300);
        });

        // Add new rows with animation
        setTimeout(() => {
            newRows.forEach((row, index) => {
                const clone = row.cloneNode(true);
                clone.style.animationDelay = `${index * 50}ms`;
                clone.classList.add('fade-in');
                currentBody.appendChild(clone);
            });
        }, 300);
    }
}

function hasDataChanged(newDoc) {
    try {
        const newChartData = JSON.parse(newDoc.getElementById('chartData').textContent);
        return JSON.stringify(chartData) !== JSON.stringify(newChartData);
    } catch (error) {
        console.error('Error comparing chart data:', error);
        return false;
    }
}

// Add realtime updates for risk scores
function setupRealtimeUpdates() {
    const alertRows = document.querySelectorAll('#alertsTable tbody tr');

    alertRows.forEach(row => {
        const riskScoreCell = row.querySelector('td:nth-child(5)');
        const currentScore = parseFloat(riskScoreCell.textContent);

        // Simulate real-time updates for demo purposes
        setInterval(() => {
            const variation = (Math.random() - 0.5) * 0.01;
            const newScore = Math.max(0, Math.min(1, currentScore + variation));

            // Update score with animation
            riskScoreCell.classList.add('score-update');
            setTimeout(() => {
                riskScoreCell.textContent = newScore.toFixed(3);
                riskScoreCell.classList.remove('score-update');
            }, 300);

            // Update status badge if risk level changes
            updateRiskLevel(row, newScore);
        }, Math.random() * 10000 + 5000);
    });
}

function updateRiskLevel(row, score) {
    const badge = row.querySelector('.status-badge');
    let newClass, newText;

    if (score > 0.85) {
        newClass = 'high-risk';
        newText = 'High Risk';
    } else if (score > 0.7) {
        newClass = 'medium-risk';
        newText = 'Medium Risk';
    } else {
        newClass = 'low-risk';
        newText = 'Low Risk';
    }

    // Update badge class and text
    badge.className = `status-badge ${newClass}`;
    badge.textContent = newText;
}

// Initialize tooltips for additional information
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(element => {
        element.addEventListener('mouseover', e => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = element.dataset.tooltip;

            document.body.appendChild(tooltip);

            const rect = element.getBoundingClientRect();
            tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;
            tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;

            element.addEventListener('mouseleave', () => tooltip.remove());
        });
    });
}

// Call initialization functions
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupAutoRefresh();
    initializeTableControls();
    setupRealtimeUpdates();
    initializeTooltips();
});