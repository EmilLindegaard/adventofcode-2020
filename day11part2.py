import numpy as np

"""
Observation: It is important to not substitute in to our current state. We must save each entry in another state matrix.
After calculating the next state, we can substitute into the current state.
"""

"""
Definitions and pre-formatting
"""

#string = "L.LL.LL.LLLLLLLLL.LLL.L.L..L..LLLL.LL.LLL.LL.LL.LLL.LLLLL.LL..L.L.....LLLLLLLLLLL.LLLLLL.LL.LLLLL.LL"
string = """LLLLLLLLLL.LLLLLLLL.LLLLLLLLLLLLL.LLLLL.LLLL.LL.L.LLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLL.L.LLLLLLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLL..LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLLLLLLLLLL..L.LLLLLLLLLLLLLLL.LL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLLLLL.LLLLL
            LLLLLLLLLL.LL..LLLL.LLLLLLL.LLLLLLLLLLL.LLLLLLLLLLL.LLLLLLLLLLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLLLL.LLLLLL.LLLL..LLLLLLLL
            LL.L..L.L..L.LL..L.....LLLL..L.L.LL..L...LLLLLL...L.....L.....LL.L...LLLLL..L.LL..L..L.LL..
            LLLLLLLLLL.LLLLLLLL.LLLLLLLL.LLLL.LLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLL..LLLLLLL.LLLLLLL.LLLLL.L.LLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLLLLLLLLLLL
            L.LLLLLLLLLL.LLL..LLLLLLLLL.LLLLL.LLLLL..LLLLLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLL..LLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLL..LLLLLLLLLLL.L.LL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLL.L.LLLLLLLLLLLLLLLL.LLLLLLLLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLL.LLLLL
            .......LL.L....L......L....L...L.LL..L..L...L.LL.L.LL..L..L.L.LLL.....L...LL..L..L..L.LL..L
            LLL.LLLLLLLLLLLLLLL.LLLLLLL.LLLLL.LLLLL.L.LL.LLLLLLLLLL.LLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLL.LLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLLLLLLLLLLLLLLLLLL..LLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLL
            LLLLLLL.LL.LLLL.LLL..LLLLLL.LLLLL.LLLLLLLLLL.LLLL.LLLLLLLLL.LLLL.LLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LL.LLLLL.LLLLLLLLLLLLL.LLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLL.LLLL.LLLL.LLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            .LL..LL.L.L....L...L.........L......LL.L..........L....L...L.L..L.L.L.LLL...LLL.L.....L.L..
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLL.LLLL.LLLLLLLLLLLLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLL.L.LLL.LLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL..LLLLL.LLLLL.LLLLLLLL
            LL.LLLLLLL.LL.LLLLL.LLLLLLL.LLLLLLLLLLLLLLLL..LLL.LLL.LLLLL.LLLLLLLL..LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLL.LL..LL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLL.L.LLLLLLLL.LL.LLLL.LLLL.LLLLLLLLL.LLL.LLLLL.L.LLLL.LLLLL.LLLLLLLL
            LL.LLLLLLL.LLLLLLLL.LLLLLLLLLLLLL.LL.LLLLLLL.LLLL..LLLLLLLLLLLLLLLL.LLLLLLLLLLLLLL.LLLLLLLL
            ......L.L......L.L.LLL.L.......L.LL..L..........LLL.LL..L.L............LL.LL.L.LL.......L..
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLL.LLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLL.LLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLLLL.LLLLLL.LLLLLLLLLLLLL.L.LLL.LLLLL.LLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLL..LLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.L.LLL.LLLLLLLLLL.LLLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLL.LLLL.LLLL.LLLLLLLLLLL.LLLLL.LLLLL.LLLL..LL..LLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLLLLL.LL
            LL.LLLLLLLLLLLLLLLL.LLL.LLL.LLLLL.L.LLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLL.L.LLLLLLLLLLLLLLLLLLL
            LLLLLL.LLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLL.LLLL.LLLLLLLLLLLLLLLLLL..LLLLLLLLL..L.LLLLLLLL
            ........LL.........L........L.L.LL..L.......L....L.....L.......L...L.L..L..........L......L
            LL.LLLLLLL.LLLLLLLL.LLLLLL.LLL.LLLLLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLLLL.LLLLLLLLLLL.LLLLL.LLLL.LLLL.LLLLLLLLLLLLLLL.LLL.LLLLLL.LLLLL.LLLLLL.L
            LLLLLLLLLL.LLLLLLLLLLLLLLLL..LLLLLLLLLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLL.LLL.LLLLLLLLLLLL.LLLLL.LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLLLLL.LLLL.LL.L.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLL.LLLLLLL.LLL.L.LLLL.LLLL.LLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            ...L....L.LL..L.......L..L...L.LL..LL.L....LL......L...L..L.....LL.LL....L.L..L.....L..L.L.
            LLLLLLLLLLLLLLLLLLLLLLLL.LL.LLLLL.LLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLLLLLLLL..LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLL.LLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLL.LLLLL.LLL.LLLLLLLLL.LLL.LL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLL.LLLLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLL.LLLL
            L.L..LL..L..LL....L...LL...L.LL...L.L.......LL.......L.LLL....L.....L..L.L..L...L.L.L.L....
            LLLLLLLLLL.LLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LL.LLLLLLLLL.LLLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLL.LLLL.LLLLLLLLLLLLLLLLLLL.LLLLLL.LLLLLLLLL.LLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLLLLLLLL.LLLLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLL.L.LLLLL.LL
            LLLLLLLLLLLLLL.LLLL..LLLLLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLL..LLLLLLLL.LLLLLL.LLL.L.LLLLLLLL
            LLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLL.LL.L.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLL.LL.LLLLLLLLLLLLL.LLLLL.LLLL.LLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            ...LLL.LL.LL.LLL......L.L.L..L..L....L........L........L...LL....L.LLL....LLL..L.......LLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLLLL.LL.LLLLLLLLL.L.LLL.LLL.LLL.LLLLLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLL.LLLLL.LLLL.LLLL.LLL.LLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLLLLL.L.LLLLLLLL..LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLL.L
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLLLLLLL.LLLL.LLLLLLLLL..LLLLLLL..LLLLLL.LLLLL.LLLLLL.L
            LLLLLLLLLL.LLLLLLLLLL.LLLLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLL.
            LLLLLLLLLL.LL.LLLLL.LLLLLLLLLLLLLLLLLLL.LLLL.LLLLLLL.LLLLLL.LLL.LLLLL.LLLLLLLLLLLLLLLLLLLLL
            LL.LL.LLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLL.LLLL.LLLLLLLLLLLLLLLL.L..LL.LLL.LLLLLLLLLLLLLL
            L.L........L...LL.L..L....LLLL...LL.L......L.L....L...L.L..LL......LL......L....L..L.LL..L.
            LLLLLLLLLLLLLLLLLLL..L.LLLL.L.LLLLLLLLL.LLLLLLLLL.LLLLLLLLL..LLLLLL.L.L..LLL.LLLLL.LLLLLLLL
            LLLLLLLL.L.LLLL.LLL.LLLLLLLLLLLLL.LLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLLLL.LL.L.LLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLLL.L
            LLLLLLLLL..LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLLLLL.L.LLLL.LLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLL.LLLLL.LLLLLLLLLLLLLLLLLLLLLLLLL.LLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLL.LL.LLLLL.LLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLL..L.LLLLLLL.LLLLL.LLL.L.LLLL..LLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.LLLLLLLLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            ...LL......L..LL.....L.LLLLLL..LL.LLL...L.L...LLL.LL........L......LLL..L.L.L.L..L..LLL.L..
            LL.LLL.LLL.LLLLLLLL.L.LLLLL.LLLLL.L.LLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.LLLLLLLLLLLLLLLL.LLLL.LLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLL.L.LL.LLLL..LLLLL.LL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLL.LLLLL..LLLLLLLL.LLLLL.L.LLLLL.LLLLL.LLL.LLLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LL.LLLL.
            LLLLLLLLLLLLLLLLLLL.LLL.LLL.LLLL.LLLLLL.LL.L.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            .......LLL....LLL.LL.L.L.L....LLL..L..L...L.LL.LLLL...LL............LL..LL....L..L...L.LLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLL.LLL
            LLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLLLLLLLLLLLL.LLLLL.LLLL.LLLLLL..L.LLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLL.LLLLL.LLLLLLLLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLL.LLLL.LLLLLLLL.LLLLLLL..LLLL.LLLLLLLLLL.LLLL.LLLLLLLLL.LLL.LLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLLLLLLLL.LLLLLLLLLLLLLLLLLLLL.LLLL.LLLLLLLLL.LLLLLLLLLLL..LLLL.LLL
            LLLLLLLLLL.LLLLLLL..LLLLLLL.LLLLL..LLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLL.LLLLLLLLL.LLLLLLL.LLLL..LLLLL.LLLL.LLLL.LLLLLLLLL.L.LLLLLLL.LLLLLLLLLLLL.L.LLLLLL
            LLLLLLLLLL.LLLLL.L..LLLLLLL.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            .....L...L..L.L..L.LLLLLLLLL....L.LL...LL...L.LL..L..L.L..L.L.....L...L.LL...L.L.......L..L
            LLLLLLLLLLLLLLLLLLL.LLLLL.L.LLLLL.LLLLL.LLLL.LLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLLLLLLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLL.LLLLLLL.LLLLL.LLLLL.LLLL.LL.LLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLLLLLLLLLLL.LLL.LLLLL.LLL.L.LLLL.LLLLLLLLLLLLLL.LLLLLLLLLLLLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL.LLLLL.LLLL.LLLL.LLLLLLLLL.LLLLLLLLL.LLLLLLLLLLLL.LLLLLLLL
            LLLLLLLL.L.LLLLLLLL.LLLLLLL.L.LLL.LLLLL.LLLLLLLLLLLLLLLLLLL.LLLLLLLLL.LLLLLL.LLLLL.LLLLLLLL
            LLLLLLLLLL.LLL.LLLL.LLLLLLL.LL.LL.LLLLL.LLLL.LLLL.LLLLLLLLLLLLLLLLLLL.LLLLLL.LLLLLLLLLLLLL.
            LLLLLLLLLL.LLLLLLL..LLLLLLL.LLLLL.LLLLL.LLLLLLLLL.LLL.LLLLL.LLLLLLLLL.LLLLLLLLLLLLLLLLLLLLL
            LLLLLLLLLL.LLLLLLLL.LLLLLLLLLLLLL.LLLLL.LLL..LLLL.LLLLLLL.L.LLLLLLLLL..LLLLLLLLLLLLLLLLLLL."""


string = "".join(string.split())
current_state = np.array(list(string)).reshape((98,91))
#current_state = np.array(list(string)).reshape((10,10))
next_state = np.empty(current_state.shape, dtype = str)

"""
Function for getting visible seats (New rule)
"""

def getVisibleSeats(current_state,next_state,row,column): #New rule
    occupied_seats = 0
    free_seats = 0
    if (row != current_state.shape[0]-1):
        for i in range(row+1,current_state.shape[0]): #Looking down from seat
            seat = current_state[i,column]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break
    if (row != 0):
        for i in reversed(range(0,row)): #Looking up from seat
            seat = current_state[i,column]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break
            
    if (column != current_state.shape[1]-1):
        for i in range(column+1,current_state.shape[1]): #Looking to the right
            seat = current_state[row,i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break
            
    if (column != 0):
        for i in reversed(range(0,column)): #Looking to the left
            seat = current_state[row,i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break

    #distances to walls
    dx_right = (current_state.shape[1]-1)-column
    dy_down = (current_state.shape[0]-1)-row
    dx_left = column
    dy_up = row

    if (row != current_state.shape[0]-1 and column != current_state.shape[1]-1):
        for i in range(1,min(dx_right,dy_down)+1): #Looking down right
            seat = current_state[row+i,column+i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break
    if (row != 0 and column != 0):
        for i in range(1,min(dx_left,dy_up)+1): #Looking up left
            seat = current_state[row-i,column-i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break    
    if (row != 0 and column != current_state.shape[1]-1):
        for i in range(1,min(dx_right,dy_up)+1): #Looking up right
            seat = current_state[row-i,column+i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break
    if (row != current_state.shape[0]-1 and column != 0):
        for i in range(1,min(dx_left,dy_down)+1): #Looking down left
            #print("Looking from: ", row,column, ". Looking down left :", dx_left,dy_down)
            seat = current_state[row+i,column-i]
            if (seat == "#"):
                occupied_seats += 1
                break
            elif (seat == "L"):
                free_seats += 1
                break

    return occupied_seats, free_seats

    
"""
Looping through matrix and substituting.
"""

for iteration in range(1000):
    print("Iteration: ", iteration+1)
    for row in range(current_state.shape[0]):
        for column in range(current_state.shape[1]):
            current_entry = current_state[row,column]
            if (current_entry == "."): #We apply rule 3
                next_state[row,column] = current_state[row,column]
            else: #We apply the new rule
                occupied_seats, free_seats = getVisibleSeats(current_state,next_state,row,column)
                #print("Entry:(",row,column,") sees: ","#:",occupied_seats,", L:",free_seats)
                if (current_entry == "#" and occupied_seats >= 5):
                    next_state[row,column] = "L"
                elif (current_entry == "L" and occupied_seats == 0):
                    next_state[row,column] = "#"
                else:
                    next_state[row,column] = current_state[row,column]

    if (np.array_equal(current_state,next_state) == 1):
        print("Final state:\n",current_state)
        unique, counts = np.unique(current_state, return_counts=True) 
        dicted_adjacents = dict(zip(unique, counts))
        print("\nNumber of occupied seats: ", dicted_adjacents['#'])
        break

    #print("CURRENT_STATE:\n",current_state, ".\nNEXT_STATE:\n", next_state)
    
    current_state = next_state

    #print("CHANGED CURRENT_STATE:\n",current_state)
    next_state = np.empty(current_state.shape, dtype = str)
    
    

    
