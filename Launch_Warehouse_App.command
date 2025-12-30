#!/bin/bash
# Move to the folder where this script is located
cd -- "$(dirname "$0")"

# Print location for troubleshooting
echo "Currently running from: $(pwd)"
echo "Looking for warehouse_app.py..."

# Run the app
python3 -m streamlit run warehouse_app.py