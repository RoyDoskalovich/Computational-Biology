import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

gridSize = 80


def updateGrid(grid):
    newGrid = grid.copy()
    for i in range(gridSize):
        for j in range(gridSize):

            verticalNeighborsCount = grid[(i - 1) % gridSize, j] + grid[(i + 1) % gridSize, j]

            leftNeighborsCount = grid[(i - 1) % gridSize, (j - 1) % gridSize] + grid[i, (j - 1) % gridSize] + grid[
                (i + 1) % gridSize, (j - 1) % gridSize]

            rightNeighborsCount = grid[(i - 1) % gridSize, (j + 1) % gridSize] + grid[i, (j + 1) % gridSize] + grid[
                (i + 1) % gridSize, (j + 1) % gridSize]

            # By saying horizontal I mean the three neighbors on the left column and the three on the right column.
            horizontalNeighborsCount = leftNeighborsCount + rightNeighborsCount

            if horizontalNeighborsCount >= 4 and verticalNeighborsCount <= 1:
                newGrid[i, j] = 0
            elif horizontalNeighborsCount < 3 and verticalNeighborsCount >= 1:
                newGrid[i, j] = 1
            elif horizontalNeighborsCount == 3 and verticalNeighborsCount > 1:
                newGrid[i, j] = 0
            elif horizontalNeighborsCount == 3 and verticalNeighborsCount < 1:
                newGrid[i, j] = 1
            else:
                newGrid[i, j] = 0 if np.random.rand() < 0.5 else 1

    return newGrid


# Evaluates how closely a given grid matches two ideal stripe patterns.
def stripesRank(grid):
    perfectPattern1 = np.tile([0, 1], (gridSize, gridSize // 2))
    perfectPattern2 = np.tile([1, 0], (gridSize, gridSize // 2))
    match1 = np.sum(grid == perfectPattern1)
    match2 = np.sum(grid == perfectPattern2)
    return max(match1, match2) / (gridSize * gridSize)


class CellularAutomatonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cellular Automaton - Vertical Stripe Formation")

        self.iterations = tk.IntVar(value=100)
        self.iterationCount = 0

        # Initialize the grid before creating widgets
        self.grid = np.random.choice([0, 1], size=(gridSize, gridSize))

        self.rankHistory = []

        self.running = False

        self.createWidgets()

    # Creates and arranges the widgets for the user interface.
    def createWidgets(self):
        controlFrame = ttk.Frame(self.root)
        controlFrame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controlFrame, text="Iterations:").pack(side=tk.LEFT, padx=5, pady=5)
        self.iterationEntry = ttk.Entry(controlFrame, textvariable=self.iterations, width=5)
        self.iterationEntry.pack(side=tk.LEFT, padx=5, pady=5)

        self.startButton = ttk.Button(controlFrame, text="Start", command=self.startSimulation)
        self.startButton.pack(side=tk.LEFT, padx=5, pady=5)

        self.stopButton = ttk.Button(controlFrame, text="Stop", command=self.startSimulation)
        self.stopButton.pack(side=tk.LEFT, padx=5, pady=5)

        self.rankLabel = ttk.Label(controlFrame, text="Rank: 0.00")
        self.rankLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.plotButton = ttk.Button(controlFrame, text="Plot Ranks", command=self.plotRanks)
        self.plotButton.pack(side=tk.LEFT, padx=5, pady=5)

        fig, self.ax = plt.subplots()
        self.mat = self.ax.matshow(self.grid, cmap='gray')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def updateSimulation(self):
        if self.running and self.iterationCount < self.iterations.get():
            self.grid = updateGrid(self.grid)
            self.mat.set_data(self.grid)
            self.canvas.draw()

            rank = stripesRank(self.grid)
            self.rankLabel.config(text=f"rank: {rank:.2f}")

            if self.iterationCount % 25 == 0:
                self.rankHistory.append(rank)

            self.iterationCount += 1
            self.root.after(50, self.updateSimulation)
        else:
            self.running = False
            self.startButton.config(state=tk.NORMAL)
            self.stopButton.config(state=tk.DISABLED)

    def startSimulation(self):
        self.running = True
        self.iterationCount = 0
        self.rankHistory = []
        self.startButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self.updateSimulation()

    def stopSimulation(self):
        self.running = False
        self.startButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.DISABLED)

    def plotRanks(self):
        if self.rankHistory:
            plt.figure()
            plt.plot(range(0, self.iterationCount, 25), self.rankHistory, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Rank')
            plt.title('Rank Over Iterations')
            plt.grid(True)
            plt.show()


root = tk.Tk()
app = CellularAutomatonApp(root)
root.mainloop()
