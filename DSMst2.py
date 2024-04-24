import numpy as np
import matplotlib.pyplot as plt

def draw_structure(node_coord, elem_con, suppress_dof):
    """
    Draw the structure with nodes, elements, and arrows representing degrees of freedom.

    Parameters:
    - node_coord (numpy.ndarray): Array containing node coordinates.
    - elem_con (numpy.ndarray): Array containing element connectivity.
    - suppress_dof (numpy.ndarray): Array containing suppressed degrees of freedom.
    """
    # Plotting the structure
    plt.figure(figsize=(8, 6))

    # Plotting nodes
    for i, (x_coord, y_coord) in enumerate(node_coord):
        plt.plot(x_coord, y_coord, 'ko')  # Plotting nodes as black circles
        plt.text(x_coord, y_coord, f'Node {i+1}', fontsize=12, ha='right', va='bottom')

    # Plotting elements
    for i, (node1, node2) in enumerate(elem_con):
        x1, y1 = node_coord[node1]
        x2, y2 = node_coord[node2]
        plt.plot([x1, x2], [y1, y2], 'b-')  # Plotting elements as blue lines

    # Plot arrows for degrees of freedom
    for i in range(len(node_coord)):
        # Plot green arrows for degrees of freedom
        plt.arrow(node_coord[i, 0], node_coord[i, 1], 0.2, 0, color='green', head_width=0.1, zorder=10)
        plt.arrow(node_coord[i, 0], node_coord[i, 1], 0, 0.2, color='green', head_width=0.1, zorder=10)

    # Plot arrows for restricted degrees of freedom
    for dof in suppress_dof:
        node_index = dof // 2
        x = node_coord[node_index, 0]
        y = node_coord[node_index, 1]
        if dof % 2 == 0:  # x-displacement
            plt.arrow(x, y, 0.2, 0, color='red', head_width=0.1, zorder=10)
        else:  # y-displacement
            plt.arrow(x, y, 0, 0.2, color='red', head_width=0.1, zorder=10)

    plt.title('Structure with Degrees of Freedom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()



# constants
E=1
A=1

# structure geometry 
node_coord=np.array([[0,0],[3,0], [3,4]]) # nodes coordinates
elem_con=np.array([[0,1],[0,2] ]) # elements connectivity

x=node_coord[:,0] # sliced array with only x node coordinates
y=node_coord[:,1] # sliced array with only y node coordinates
node_count=len(node_coord)
elem_count=len(elem_con)
struct_dof=2*node_count # (entire) structure degrees of freedom

# matrices initialization
displacement=np.zeros((struct_dof,1))
force=np.zeros((struct_dof,1))
sigma=np.zeros((elem_count,1))
stiffness=np.zeros((struct_dof,struct_dof))

# constrained degrees of freedom
suppress_dof=np.array([2,3,4,5]) 

# load assignments
force[1] = -2   

# known forces and displacements
for i in range(len(suppress_dof)):
    displacement[suppress_dof[i]]=0
    force[suppress_dof[i]]=0
membersStifness = [None] * elem_count


for e in range(elem_count):
    index = elem_con[e]
    elem_dof = np.array([index[0]*2, index[0]*2+1, index[1]*2, index[1]*2+1])
    xl = x[index[1]] - x[index[0]]   
    yl = y[index[1]] - y[index[0]]   
    elem_length = np.sqrt(xl * xl + yl * yl)   
    c = xl / elem_length   
    s = yl / elem_length      
    print(f"For Element {e+1} (Index {index}):")
    print("  elem_dof =", elem_dof)
    print(" element Length ", elem_length)
    print(" lamda X ", c)
    print(" lamda Y ",s)
    rot = np.array([[c*c, c*s, -c*c, -c*s],
                    [c*s, s*s, -c*s, -s*s],
                    [-c*c, -c*s, c*c, c*s],
                    [-c*s, -s*s, c*s, s*s]])
    k = (E * A / elem_length) * rot
    membersStifness[e] = k.tolist()
    print(" Stifness ")
    print(k)
    # Truss Stiffness Matrix 
    for i in range(4):
        for j in range(4):
            stiffness[elem_dof[i], elem_dof[j]] += k[i, j]
    
print(" Truss Stifness Matrix") 
print(stiffness)   
print(" Member  Stifness Matrix") 
print(membersStifness)  
print(" Displacement ")
print(displacement)
print(" Force ") 
print(force)
 

draw_structure(node_coord, elem_con, suppress_dof)
active_dof=np.setdiff1d(np.arange(struct_dof), suppress_dof)
displacement_aux=np.linalg.solve(stiffness[np.ix_(active_dof,active_dof)], force[np.ix_(active_dof)])
displacement[np.ix_(active_dof)]=displacement_aux
react=np.dot(stiffness, displacement)

# emitting results to screen
print(f'displacements:\n {displacement}')
print(f'Force:\n {react}')