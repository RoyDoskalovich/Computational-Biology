# Cellular Automaton for Vertical Stripe Formation

## Overview
This implementation uses a cellular automaton to simulate the formation of vertical stripe patterns in a grid. The program features a graphical interface that allows users to visualize the evolution of the patterns and monitor the system's progress toward stripe formation.

## Approach
The solution employs a custom cellular automaton with specific rules:

### Cell State Rules
- Each cell can be either black (1) or white (0)
- State transitions depend on both horizontal and vertical neighbors
- The grid is toroidal (wraps around at edges)

### Neighbor Counting
- **Vertical Neighbors**: Counts adjacent cells in the same column
  - Maximum: 2 neighbors
  - Minimum: 0 neighbors

- **Horizontal Neighbors**: Counts cells in adjacent columns
  - Left column: Up to 3 neighbors
  - Right column: Up to 3 neighbors
  - Maximum total: 6 neighbors
  - Minimum total: 0 neighbors

### State Transition Rules
1. If horizontal neighbors ≥ 4 and vertical neighbors ≤ 1:
   - Cell becomes white (0)
2. If horizontal neighbors < 3 and vertical neighbors ≥ 1:
   - Cell becomes black (1)
3. If horizontal neighbors = 3:
   - With vertical neighbors > 1: Cell becomes white (0)
   - With vertical neighbors < 1: Cell becomes black (1)
4. In all other cases:
   - Random state assignment with 50/50 probability

## Features
- Interactive GUI using tkinter
- Real-time visualization using matplotlib
- Pattern quality evaluation through stripe ranking
- Progress tracking and visualization
- Configurable iteration count
- Start/Stop simulation control
- Rank history plotting

## Implementation Details

### Parameters
- Grid Size: 80x80 cells
- Iteration control via GUI
- Update interval: 50ms
- Rank calculation every 25 iterations

### Components
1. **Grid Management**:
   - Toroidal grid implementation
   - Efficient neighbor counting
   - State transition logic

2. **Visualization**:
   - Real-time grid display
   - Grayscale color mapping
   - Rank history plotting

3. **Pattern Evaluation**:
   - Stripe pattern matching
   - Normalized ranking system
   - Progress tracking

### Interface Controls
- **Start Button**: Begins/resumes simulation
- **Stop Button**: Pauses simulation
- **Iterations Input**: Sets maximum iterations
- **Plot Ranks**: Displays rank history graph
- **Rank Display**: Shows current pattern quality

## Usage
To run the simulation:
1. Ensure Python 3.x with numpy, matplotlib, and tkinter is installed
2. Download the CA.exe file from this [drive](https://drive.google.com/drive/folders/1YV0cZw733MaLh9Rk8jVq1_u4qel_gbZU?usp=sharing)
3. Run the CA.exe
4. Use the GUI controls to:
   - Set desired iteration count
   - Start/stop the simulation
   - Monitor pattern formation
   - View rank history

The program will display the evolving grid pattern and provide real-time feedback on stripe formation quality through the rank metric.
