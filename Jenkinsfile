pipeline {
    agent any

    environment {
        CODE_DIRECTORY = 'recmodel'
        CUDA_VISIBLE_DEVICES = '0'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout SCM') {
            steps {
                script {
                    // Checkout code
                    checkout scm
                }
            }
        }

//         stage('Check Code') {
//             steps {
//                 script {
//                     // Check line count and changes
//                     sh '''
//                     echo "=== Checking lines of code in each file ==="
//                     MAX_LINES=500
//                     set +x
//                     git ls-files | grep -Ev ".pylintrc|airflow.cfg|data" | while read -r file; do
//                         line_count=$(wc -l < "$file")
//                         if [ "$line_count" -gt "$MAX_LINES" ]; then
//                             echo "Error: File $file has $line_count lines, which exceeds the threshold of $MAX_LINES lines."
//                             exit 1
//                         fi
//                     done
//
//                     echo "=== Checking lines of code changes ==="
//                     MAX_CHANGE_LINES=300
//                     git fetch origin main
//                     LAST_MAIN_COMMIT=$(git rev-parse origin/main)
//                     CURRENT_COMMIT=$(git rev-parse HEAD)
//
//                     if [ "$LAST_MAIN_COMMIT" = "$CURRENT_COMMIT" ]; then
//                         DIFF_RANGE="HEAD^1..HEAD"
//                     else
//                         DIFF_RANGE="origin/main..HEAD"
//                     fi
//
//                     CHANGES=$(git diff --numstat "$DIFF_RANGE" | awk '{added+=$1; deleted+=$2} END {print added+deleted}')
//                     if [ -n "$CHANGES" ] && [ "$CHANGES" -gt "$MAX_CHANGE_LINES" ]; then
//                         echo "Error: Too many changes: $CHANGES lines."
//                         exit 1
//                     else
//                         echo "Number of changed lines: $CHANGES"
//                     fi
//                     '''
//                 }
//             }
//         }

        stage('Setup and Run Pipeline') {
            agent {
                docker {
                    image 'ubuntu22.04_python39_cuda12.2'
                    args '--gpus all -u root'
                }
            }

            environment {
                CUDA_VISIBLE_DEVICES = "${CUDA_VISIBLE_DEVICES}"
            }

            steps {
                script {
                    // Set up Python environment once
                    sh '''
                    chmod -R 777 /var/jenkins_home/pip_cache
                    echo "=== Setting up Python environment ==="
                    python3 -m pip install --cache-dir /var/jenkins_home/pip_cache --extra-index-url https://pypi.nvidia.com -e .[dev]
                    '''
//
//                     // Run linting
//                     sh '''
//                     echo "=== Running Linting Tools ==="
//                     flake8 $CODE_DIRECTORY
//                     mypy --show-traceback $CODE_DIRECTORY
//                     '''

                    // Run tests
                    sh '''
                     echo "=== Running Tests ==="
                     rm -rf recmodel/*pycache*
                     ls -la recmodel
                     chown -R root:root .
                     pytest -s --durations=0 --disable-warnings tests/
                    '''

                    // Run main application
//                     sh '''
//                     echo "=== Running Main File ==="
//                     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 main.py
//                     '''
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}