FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy files to the container
# COPY . /app
COPY src /app/src
COPY scripts/002x_review_burst_detection.py /app/main.py
# COPY scripts/011_interactive_tsne.py /app/interactive_tsne.py
COPY requirements.txt /app/requirements.txt
COPY results/burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/df_cultures.pkl /app/results/burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/df_cultures.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for consistent paths (optional)
ENV RESULTS_FOLDER=/app/results
ENV DEBUG=False

# Expose the port Dash runs on
EXPOSE 8080

# Run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:server"]
