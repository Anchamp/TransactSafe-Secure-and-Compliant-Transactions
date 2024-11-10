document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const fileInput = document.getElementById('file');
    const fileInfo = document.querySelector('.file-info');
    const submitButton = predictionForm.querySelector('button[type="submit"]');
    const spinner = submitButton.querySelector('.spinner');
    const buttonText = submitButton.querySelector('.btn-text');

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Validate file type
            if (!file.name.toLowerCase().endsWith('.csv')) {
                showError('Please select a CSV file');
                fileInput.value = '';
                fileInfo.textContent = '';
                return;
            }

            // Validate file size (max 4GB)
            const maxSize = 4 * 1024 * 1024 * 1024;
            if (file.size > maxSize) {
                showError('File size exceeds 4GB limit');
                fileInput.value = '';
                fileInfo.textContent = '';
                return;
            }

            // Display file info
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            fileInfo.textContent = `Selected file: ${file.name} (${sizeMB} MB)`;
        } else {
            fileInfo.textContent = '';
        }
    });

    // Form submission handler
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            showError('Please select a file');
            return;
        }

        // Start loading state
        setLoading(true);
        showProgressSection();

        const formData = new FormData(predictionForm);

        try {
            const response = await fetch(predictionForm.action, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.status === 'processing') {
                // Start checking processing status
                startProcessingCheck();
            } else if (data.status === 'error') {
                showError(data.message);
                setLoading(false);
            } else {
                // Refresh page to show results
                window.location.reload();
            }

        } catch (error) {
            console.error('Error:', error);
            showError('Error uploading file. Please try again.');
            setLoading(false);
        }
    });

    // Handle table sorting
    document.querySelectorAll('#resultsTable th').forEach(headerCell => {
        headerCell.addEventListener('click', () => {
            const tableBody = headerCell.closest('table').querySelector('tbody');
            const rows = Array.from(tableBody.querySelectorAll('tr'));
            const columnIndex = headerCell.cellIndex;
            const isNumeric = !isNaN(rows[0].children[columnIndex].textContent.trim());

            const sortedRows = sortTableRows(rows, columnIndex, isNumeric);

            // Clear table body
            while (tableBody.firstChild) {
                tableBody.removeChild(tableBody.firstChild);
            }

            // Add sorted rows with animation
            sortedRows.forEach((row, index) => {
                row.style.animationDelay = `${index * 50}ms`;
                tableBody.appendChild(row);
            });
        });
    });

    function sortTableRows(rows, columnIndex, isNumeric) {
        return rows.sort((rowA, rowB) => {
            const cellA = rowA.children[columnIndex].textContent.trim();
            const cellB = rowB.children[columnIndex].textContent.trim();

            if (isNumeric) {
                // Handle currency values
                const numA = parseFloat(cellA.replace(/[$,]/g, ''));
                const numB = parseFloat(cellB.replace(/[$,]/g, ''));
                return numA - numB;
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

    function showError(message) {
        const alertElement = document.createElement('div');
        alertElement.className = 'alert alert-warning';
        alertElement.textContent = message;

        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        // Insert new alert
        predictionForm.parentNode.insertBefore(alertElement, predictionForm);

        // Remove alert after 5 seconds
        setTimeout(() => alertElement.remove(), 5000);
    }

    function setLoading(isLoading) {
        submitButton.disabled = isLoading;
        spinner.style.display = isLoading ? 'block' : 'none';
        buttonText.style.display = isLoading ? 'none' : 'block';
        fileInput.disabled = isLoading;
    }

    function showProgressSection() {
        const progressSection = document.createElement('div');
        progressSection.className = 'progress-section';
        progressSection.innerHTML = `
            <div id="progressStatus">Processing file...</div>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill"></div>
            </div>
            <div id="progressPercentage">0%</div>
        `;

        predictionForm.insertAdjacentElement('afterend', progressSection);
        progressSection.style.display = 'block';let processingCheckInterval;

    async function startProcessingCheck() {
        processingCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/processing_status');
                const data = await response.json();

                if (data.is_processing) {
                    updateProgress(data);
                } else {
                    // Processing complete
                    clearInterval(processingCheckInterval);
                    setLoading(false);
                    completeProgress();

                    if (data.error) {
                        showError(data.error);
                    } else {
                        // Reload page to show results
                        setTimeout(() => {
                            window.location.reload();
                        }, 1000);
                    }
                }
            } catch (error) {
                console.error('Error checking status:', error);
                clearInterval(processingCheckInterval);
                setLoading(false);
                showError('Error checking processing status');
            }
        }, 1000);
    }

    function updateProgress(data) {
        const progressStatus = document.getElementById('progressStatus');
        const progressFill = document.getElementById('progressFill');
        const progressPercentage = document.getElementById('progressPercentage');

        if (progressStatus && progressFill && progressPercentage) {
            const progress = data.total_chunks > 0
                ? (data.current_chunk / data.total_chunks) * 100
                : 0;

            progressFill.style.width = `${progress}%`;
            progressPercentage.textContent = `${Math.round(progress)}%`;
            progressStatus.textContent = data.status_message || 'Processing file...';
        }
    }

    function completeProgress() {
        const progressStatus = document.getElementById('progressStatus');
        const progressFill = document.getElementById('progressFill');
        const progressPercentage = document.getElementById('progressPercentage');

        if (progressStatus && progressFill && progressPercentage) {
            progressFill.style.width = '100%';
            progressPercentage.textContent = '100%';
            progressStatus.textContent = 'Processing complete!';
        }
    }

    // Results table enhancements
    if (document.getElementById('resultsTable')) {
        initializeResultsTable();
    }

    function initializeResultsTable() {
        // Add risk highlighting
        const rows = document.querySelectorAll('#resultsTable tbody tr');
        rows.forEach(row => {
            const riskScore = parseFloat(row.querySelector('td:nth-child(6)').textContent);
            if (riskScore > 0.85) {
                row.classList.add('high-risk');
            } else if (riskScore > 0.7) {
                row.classList.add('medium-risk');
            }
        });

        // Add export functionality
        document.querySelector('.btn-secondary')?.addEventListener('click', exportResults);
    }

    function exportResults() {
        const table = document.getElementById('resultsTable');
        if (!table) return;

        try {
            // Create CSV content
            const headers = Array.from(table.querySelectorAll('th'))
                .map(th => `"${th.textContent}"`);

            const rows = Array.from(table.querySelectorAll('tbody tr'))
                .map(row => Array.from(row.querySelectorAll('td'))
                    .map(td => `"${td.textContent.trim()}"`)
                    .join(',')
                );

            const csvContent = [headers.join(','), ...rows].join('\n');

            // Create download link
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `transaction_predictions_${new Date().toISOString().split('T')[0]}.csv`;

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

        } catch (error) {
            console.error('Error exporting results:', error);
            showError('Error exporting results. Please try again.');
        }
    }

    // Add summary card animations
    function initializeSummaryAnimations() {
        const summaryValues = document.querySelectorAll('.summary-item .value');

        summaryValues.forEach(value => {
            const targetValue = parseFloat(value.textContent.replace(/[^0-9.-]+/g, ''));
            animateValue(value, 0, targetValue, 1000);
        });
    }

    function animateValue(element, start, end, duration) {
        const startTimestamp = performance.now();
        const isPercentage = element.textContent.includes('%');
        const isCurrency = element.textContent.includes('$');

        const animate = (currentTimestamp) => {
            const elapsed = currentTimestamp - startTimestamp;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = start + (end - start) * easeOutQuart;

            // Format based on type
            if (isCurrency) {
                element.textContent = `$${current.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })}`;
            } else if (isPercentage) {
                element.textContent = `${current.toFixed(1)}%`;
            } else {
                element.textContent = Math.round(current).toLocaleString('en-US');
            }

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Initialize tooltips for risk scores
    function initializeTooltips() {
        const riskScores = document.querySelectorAll('#resultsTable td:nth-child(6)');

        riskScores.forEach(cell => {
            const score = parseFloat(cell.textContent);
            let riskLevel = 'Low Risk';
            let description = 'This transaction appears normal.';

            if (score > 0.85) {
                riskLevel = 'High Risk';
                description = 'Immediate review recommended.';
            } else if (score > 0.7) {
                riskLevel = 'Medium Risk';
                description = 'Further investigation may be needed.';
            }

            cell.setAttribute('title', `${riskLevel}: ${description}`);
            cell.style.cursor = 'help';
        });
    }

    // Call initialization functions if results are present
    if (document.querySelector('.results-section')) {
        initializeSummaryAnimations();
        initializeTooltips();
    }
});