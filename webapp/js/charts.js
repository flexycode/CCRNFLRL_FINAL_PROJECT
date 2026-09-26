/**
 * charts.js — Interactive Chart.js visualizations for training/benchmark results.
 */

const Charts = (() => {
  // Shared chart options for dark theme
  const darkTheme = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        labels: { color: '#a0a0b8', font: { family: 'Inter', size: 12 } }
      },
      tooltip: {
        backgroundColor: 'rgba(18, 18, 26, 0.95)',
        titleColor: '#f0f0f5',
        bodyColor: '#a0a0b8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        titleFont: { family: 'Inter', weight: 'bold' },
        bodyFont: { family: 'Inter' }
      }
    },
    scales: {
      x: {
        ticks: { color: '#6a6a82', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' }
      },
      y: {
        ticks: { color: '#6a6a82', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.06)' }
      }
    }
  };

  /**
   * Create the training curves chart (win rate + avg guesses).
   */
  function createTrainingChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.episodes,
        datasets: [
          {
            label: 'Win Rate',
            data: data.win_rates.map(v => v * 100),
            borderColor: '#34d399',
            backgroundColor: 'rgba(52, 211, 153, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5,
            borderWidth: 2,
            yAxisID: 'y'
          },
          {
            label: 'Avg Guesses',
            data: data.avg_guesses,
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96, 165, 250, 0.05)',
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5,
            borderWidth: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        ...darkTheme,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          ...darkTheme.plugins,
          title: {
            display: true,
            text: `Training Progress (${data.total_episodes.toLocaleString()} Episodes, ${data.n_words.toLocaleString()} Words)`,
            color: '#f0f0f5',
            font: { family: 'Inter', size: 14, weight: 'bold' }
          },
          annotation: data.curriculum_stages ? {
            annotations: createCurriculumAnnotations(data)
          } : undefined
        },
        scales: {
          x: {
            ...darkTheme.scales.x,
            title: { display: true, text: 'Episode', color: '#6a6a82', font: { family: 'Inter' } }
          },
          y: {
            ...darkTheme.scales.y,
            position: 'left',
            title: { display: true, text: 'Win Rate (%)', color: '#34d399', font: { family: 'Inter' } },
            min: 0,
            max: 105
          },
          y1: {
            ...darkTheme.scales.y,
            position: 'right',
            title: { display: true, text: 'Avg Guesses', color: '#60a5fa', font: { family: 'Inter' } },
            min: 2,
            max: 6,
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  }

  function createCurriculumAnnotations(data) {
    if (!data.curriculum_stages || data.curriculum_stages.length <= 1) return {};
    const annotations = {};
    const totalEp = data.total_episodes;
    const stages = data.curriculum_stages;
    const colors = ['#a78bfa', '#f472b6', '#fb923c', '#34d399', '#60a5fa', '#fbbf24', '#f87171'];

    for (let i = 1; i < stages.length; i++) {
      const epFrac = i / stages.length;
      const ep = Math.round(epFrac * totalEp);
      annotations[`stage${i}`] = {
        type: 'line',
        xMin: ep,
        xMax: ep,
        borderColor: colors[i % colors.length] + '60',
        borderWidth: 1,
        borderDash: [4, 4],
        label: {
          display: true,
          content: `→ ${stages[i]} words`,
          position: 'start',
          color: colors[i % colors.length],
          font: { size: 10, family: 'Inter' },
          backgroundColor: 'rgba(0,0,0,0.6)',
          padding: 4
        }
      };
    }
    return annotations;
  }

  /**
   * Create the loss chart.
   */
  function createLossChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.episodes,
        datasets: [{
          label: 'Training Loss',
          data: data.losses,
          borderColor: '#fb923c',
          backgroundColor: 'rgba(251, 146, 60, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 5,
          borderWidth: 2
        }]
      },
      options: {
        ...darkTheme,
        plugins: {
          ...darkTheme.plugins,
          title: {
            display: true,
            text: 'Training Loss',
            color: '#f0f0f5',
            font: { family: 'Inter', size: 14, weight: 'bold' }
          }
        },
        scales: {
          x: {
            ...darkTheme.scales.x,
            title: { display: true, text: 'Episode', color: '#6a6a82', font: { family: 'Inter' } }
          },
          y: {
            ...darkTheme.scales.y,
            title: { display: true, text: 'Loss', color: '#fb923c', font: { family: 'Inter' } }
          }
        }
      }
    });
  }

  /**
   * Create a guess distribution bar chart.
   */
  function createDistChart(canvasId, distribution, label = 'Guess Distribution') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const labels = ['1', '2', '3', '4', '5', '6'];
    const values = labels.map(l => distribution[l] || 0);
    const max = Math.max(...values, 1);
    const colors = values.map((_, i) => {
      const hue = 120 + i * 40;
      return `hsla(${hue}, 60%, 50%, 0.8)`;
    });

    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: label,
          data: values,
          backgroundColor: colors,
          borderColor: colors.map(c => c.replace('0.8', '1')),
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        ...darkTheme,
        indexAxis: 'y',
        plugins: {
          ...darkTheme.plugins,
          legend: { display: false },
          title: {
            display: true,
            text: label,
            color: '#f0f0f5',
            font: { family: 'Inter', size: 14, weight: 'bold' }
          }
        },
        scales: {
          x: {
            ...darkTheme.scales.x,
            title: { display: true, text: 'Count', color: '#6a6a82', font: { family: 'Inter' } }
          },
          y: {
            ...darkTheme.scales.y,
            title: { display: true, text: 'Guesses', color: '#6a6a82', font: { family: 'Inter' } }
          }
        }
      }
    });
  }

  /**
   * Create a comparison radar chart.
   */
  function createComparisonRadar(canvasId, dqnData, heuristicData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Win Rate', 'Avg Guesses (inv)', 'Generalization', 'Speed', 'Consistency'],
        datasets: [
          {
            label: 'DQN Agent',
            data: [
              dqnData.win_rate * 100,
              (7 - dqnData.avg_guesses) / 7 * 100,
              70,
              85,
              80
            ],
            borderColor: '#a78bfa',
            backgroundColor: 'rgba(167, 139, 250, 0.15)',
            borderWidth: 2,
            pointBackgroundColor: '#a78bfa'
          },
          {
            label: 'Entropy Heuristic',
            data: [
              heuristicData.win_rate * 100,
              (7 - heuristicData.avg_guesses) / 7 * 100,
              50,
              40,
              100
            ],
            borderColor: '#34d399',
            backgroundColor: 'rgba(52, 211, 153, 0.15)',
            borderWidth: 2,
            pointBackgroundColor: '#34d399'
          }
        ]
      },
      options: {
        ...darkTheme,
        plugins: {
          ...darkTheme.plugins,
          title: {
            display: true,
            text: 'Agent Comparison',
            color: '#f0f0f5',
            font: { family: 'Inter', size: 14, weight: 'bold' }
          }
        },
        scales: {
          r: {
            grid: { color: 'rgba(255,255,255,0.06)' },
            angleLines: { color: 'rgba(255,255,255,0.06)' },
            ticks: { display: false },
            pointLabels: { color: '#a0a0b8', font: { family: 'Inter', size: 11 } },
            suggestedMin: 0,
            suggestedMax: 100
          }
        }
      }
    });
  }

  return {
    createTrainingChart,
    createLossChart,
    createDistChart,
    createComparisonRadar,
    darkTheme
  };
})();
