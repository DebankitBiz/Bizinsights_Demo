
def render_dataframe_with_tooltips(config_df):
    html_code = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSV Reader with Tooltips</title>
    <style>
        body {
            font-family: Sans serif Script, sans-serif;
            /* Set font family for the entire document */
        }

        .container {
            padding: -1rem;
            border: 5px solid #ddd;
            border-radius: 10px;
            margin: 1rem 0;
            overflow-x: auto;
            /* Allow horizontal scrolling if needed */
            overflow-y: scroll;
            /* Add vertical scrollbar */
            max-height: 400px;
        }

        table {
            border-collapse: collapse;
            width: 1000px;
            height: 1000px;
        }

        th,
        td {
            border: 1px solid #ddd;
            /* Define border for th and td */
            padding: 8px;
            font-size: 12px; /* Reduce font size for table content */
            white-space: nowrap;
            /* Prevent line breaks */
            overflow: hidden;
            /* Hide overflow content */
            text-overflow: ellipsis;
            /* Display ellipsis for overflow content */
            width: 50px; /* Set fixed width for all cells */
        }

        th {
            background-color: #f2f2f2;
            position: sticky;
            /* Makes the header row sticky */
            top: 0;
            z-index: 1;
            /* Ensures the header stays above other content */
            border-bottom: 1px solid #ddd;
            /* Ensures the bottom border is visible */
            font-weight: bold;
            /* Make header text bold */
            font-size: 14px; /* Reduce font size for header */
        }

        td {
            text-align: left;
            border-bottom: 1px solid #ddd;
            /* Ensures the bottom border is visible */
        }
        
        tr {
            height: 5px; /* Set the fixed height for table rows */
        }


        .pass {
            color: green;
        }

        .fail {
            color: red;
        }

        .Pass_with_comment {
            color: Orange;
        }

        .tooltip {
            position: fixed;
            /* Set position to fixed */
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px;
            border-radius: 5px;
            z-index: 1;
            white-space: nowrap;
            pointer-events: none;
            /* Prevent tooltip from blocking hover events */
        }
    </style>
</head>

<body>
    <div>
        <label for="filterSelect">Filter by Check Column:</label>
        <select id="filterSelect" onchange="filterTable(this.value)">
            <option value="All">All</option>
            <!-- Options for unique values in the Check column will be dynamically injected here -->
        </select>
    </div>

    <div class="container">
        <table id="csvTable">
            <thead>
                <!-- Table header will be dynamically injected here -->
            </thead>
            <tbody>
                <!-- CSV Table data will be injected here -->
            </tbody>
        </table>
    </div>

    <script>
        const data = """ + config_df.to_json(orient="split") + """;

        const table = document.getElementById('csvTable');
        const filterSelect = document.getElementById('filterSelect');
        const tooltip = document.createElement('div'); // Create a single tooltip element
        tooltip.classList.add('tooltip');
        document.body.appendChild(tooltip); // Append tooltip to body

        // Create header row
        const headerRow = document.createElement('tr');
        for (const header of data.columns) {
            // Hide the Comments column
            if (header !== 'Comments') {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            }
        }
        const thead = document.createElement('thead');
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create data rows
        const tbody = document.createElement('tbody');
        for (const row of data.data) {
            const dataRow = document.createElement('tr');
            for (let i = 0; i < row.length; i++) {
                // Hide the Comments column data
                if (data.columns[i] !== 'Comments') {
                    const td = document.createElement('td');
                    td.textContent = row[i];
                    if (data.columns[i] === 'Status') {
                        // Change text color based on status
                        if (row[i] === 'Pass') {
                            td.classList.add('pass');
                        } else if (row[i] === 'Fail') {
                            td.classList.add('fail');
                        } else if (row[i] === 'Pass.') {
                            td.classList.add('Pass_with_comment')
                        } else if (row[i] === 'Pass_Null') {
                            td.classList.add('Pass_with_comment');
                        }
                        // Add tooltip if it's the Status column
                        td.setAttribute('data-tooltip', row[data.columns.indexOf('Comments')]);
                        // Add event listeners for mouseover and mouseout
                        td.addEventListener('mouseover', showTooltip);
                        td.addEventListener('mouseout', hideTooltip);
                    }
                    dataRow.appendChild(td);
                }
            }
            tbody.appendChild(dataRow);
        }
        table.appendChild(tbody);

        // Function to show tooltip
        function showTooltip(event) {
            const target = event.target;
            const tooltipText = target.getAttribute('data-tooltip');
            tooltip.textContent = tooltipText;
            tooltip.style.top = (event.pageY + 10) + 'px'; // Position tooltip below mouse pointer
            tooltip.style.left = event.pageX + 'px'; // Position tooltip at mouse pointer
            tooltip.style.display = 'block';
        }

        // Function to hide tooltip
        function hideTooltip() {
            tooltip.style.display = 'none';
        }

        // Function to filter the table based on the selected value
        function filterTable(value) {
            const rows = table.querySelectorAll('tbody tr');
            for (let row of rows) {
                const cells = row.querySelectorAll('td');
                let shouldDisplay = false;
                for (let cell of cells) {
                    if (cell.textContent === value || value === 'All') {
                        shouldDisplay = true;
                        break;
                    }
                }
                if (shouldDisplay) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }

            // Adjust row heights after filtering
            adjustRowHeights();
        }

        // Function to adjust row heights based on content
        function adjustRowHeights() {
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(row => {
                let maxHeight = 0;
                const cells = row.querySelectorAll('td');
                cells.forEach(cell => {
                    maxHeight = Math.max(maxHeight, cell.offsetHeight);
                });
                row.style.height = maxHeight + 'px';
            });
        }

        // Populate the filter select box with unique values from the Check column
        const checkColumnIndex = data.columns.indexOf('Check');
        if (checkColumnIndex !== -1) {
            const uniqueValues = new Set(data.data.map(row => row[checkColumnIndex]));
            for (const value of uniqueValues) {
                const option = document.createElement('option');
                option.textContent = value;
                option.value = value;
                filterSelect.appendChild(option);
            }
        }
    </script>
</body>

</html>







    """

    return html_code
