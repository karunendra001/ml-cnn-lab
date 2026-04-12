import numpy as np

# -----------------------------
# Convolution Function
# -----------------------------
def conv2d(image, kernel, stride=1, padding=0):
    
    # Get shapes
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Output size formula
    out_h = (img_h - k_h + 2*padding)//stride + 1
    out_w = (img_w - k_w + 2*padding)//stride + 1

    # Output feature map
    output = np.zeros((out_h, out_w))

    # Convolution operation
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Extract region
            region = image[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            
            # Element-wise multiplication
            output[i, j] = np.sum(region * kernel)

    return output

# -----------------------------
# Test Image (5x5)
# -----------------------------
image = np.array([
    [3,1,0,2,4],
    [1,5,3,2,1],
    [0,2,6,4,3],
    [2,3,1,5,2],
    [1,0,2,3,4]
])

# Sobel-X kernel
kernel = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

# -----------------------------
# Run Convolution
# -----------------------------
output = conv2d(image, kernel, stride=1, padding=0)

print("Output Feature Map:\n", output)
print("Output Shape:", output.shape)