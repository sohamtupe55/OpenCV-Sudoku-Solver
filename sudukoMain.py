import sys

import SudokuSolver
from Utils import *
import timeit
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Let's suppress the TensorFlow messages :-)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############################################################
DEBUG = False
ANSWERS_ONLY = True
# pathImage = "Resources/1.jpg"
# pathImage = "Resources/Puzzle Page Sudoku 10-20-2020.jpg"
heightImg = 900  # Originally 450x450
widthImg = 900
MURTAZA_MODEL = 1
SCOTT_MODEL = 2
model = initializePredictionModel(SCOTT_MODEL)  # Load the CNN model
############################################################

# ------------------------------------------------------------------------------
# 0. Get file
root = Tk()
root.withdraw()
pathImage = askopenfilename(title='Select Sudoku Puzzle Image')
root.update()
root.destroy()
if not pathImage:
    print("ERROR: No Sudoku Puzzle Image was provided...")
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. Prepare the image
print(f"--> Preparing the image: {pathImage}")
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # make sure the image is square
imgBlank = np.zeros((widthImg, heightImg, 3), np.uint8)  # create a blank image for testing, debugging
imgThreshold = preProcess(img)

# ------------------------------------------------------------------------------
# 2. Find all of the contours
print("--> Looking for the contours")
imgContours = img.copy()  # Keep a separate copy for display purposes
imgBigContour = img.copy()  # Used to highlight the Sudoku board
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # Draw all detected contours

# ------------------------------------------------------------------------------
# 3. Find the biggest contour as it should be the Sudoku puzzle
print("--> Looking for the Sudoku board")
biggest, maxArea = biggestContour(contours)  # Find the biggest contour (e.g. the Sudoku Board)
if biggest.size != 0:
    print("--> Found the Sudoku board!")
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)
    pts1 = np.float32(biggest)  # prepare points for Warp
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------------------
    # 4. Split the image and find each digit available
    print("--> Looking for the numbers")
    imgSolvedDigits = imgBlank.copy()
    imgSolvedDigits2 = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print("--> Identifying the numbers ;-)")
    numbers = getPrediction(boxes, model)
    imgDetectDigits = displayNumbers(imgDetectDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)  # Find all of the empty squares for which we need to find the solution

    # showStackedImage([[img, imgThreshold, imgContours, imgBigContour],
    #                   [imgDetectDigits, imgBlank, imgBlank, imgBlank]], scale=0.6)

    # ------------------------------------------------------------------------------
    # 5. Find the solution of the Sudoku board
    board = np.array_split(numbers, 9)  # splits the list of numbers into a 9x9 array
    # board2 = np.array_split(numbers, 9)
    if DEBUG:
        print(f'Board (before): {board}')
    try:
        print("--> Solving the Sudoku board!")
        starttime = timeit.default_timer()
        SudokuSolver.solve(board)
        print(f'** It took the Sudoku Solver {timeit.default_timer() - starttime:{2}.{3}}s to solve!')
    except:
        pass  # In case a solution was not found

    if DEBUG:
        print(f'Board (after): {board}')

    if DEBUG:
        starttime = timeit.default_timer()
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        print(f'flatList time: {timeit.default_timer() - starttime}')

    if DEBUG:
        starttime = timeit.default_timer()
        flatList = list(np.ndarray.flatten(np.array(board)))
        print(f'flatList (numpy flattened) time: {timeit.default_timer() - starttime}')

        print(f'flatList: {flatList}, len(flatList): {len(flatList)}')
        print(f'posArray: {posArray}, len(posArray): {len(posArray)}')
    else:
        flatList = list(np.ndarray.flatten(np.array(board)))

    solvedNumbers = flatList * posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers, (0, 0, 255))  # Print only the solved digits

    # ------------------------------------------------------------------------------
    # 6. Overlay Solution
    pts2 = np.float32(biggest)  # Prepare points for Warp
    pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpedColored = img.copy()
    imgInvWarpedColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpedColored, 1, img, 0.5, 1)
    imgDetectDigits = drawGrid(imgDetectDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    if ANSWERS_ONLY:
        showStackedImage([[cv2.resize(img, (600, 800)), cv2.resize(inv_perspective, (600, 800))]], 0.75)
    else:
        showStackedImage([[img, imgThreshold, imgContours, imgBigContour],
                         [imgDetectDigits, imgSolvedDigits, imgInvWarpedColored, inv_perspective]], scale=0.6)

    cv2.waitKey(0)
else:
    print("A Sudoku Board was not found!")
