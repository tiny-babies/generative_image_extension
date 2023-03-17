import cv2
import numpy as np


img = cv2.imread('./dataset/aquarium/places365_val_00001341.jpg')



flippedimg = cv2.flip(img, 1)

finalImg = cv2.hconcat([img, flippedimg])


height = img.shape[0]
width = img.shape[1]




# finalImg[:, width:width + int(width / 4)] = 255

finalImg = finalImg[:, width - int(width / 2):width + int(width/2)]

finalImg[:, int(width/2):int(width/2)+int(width/4)] = 255

mask = np.zeros(finalImg.shape[:2], dtype="uint8")
mask[:, int(width/2):int(width/2)+int(width/4)] = 255


print(mask.shape)
print(finalImg.shape)


# tmpMask = np.empty((height, int(width / 4), 3), dtype="uint8")
# tmpMask.fill(255)

# empty = np.zeros((height, width, 3), dtype="uint8")

print(empty.shape)

cv2.imwrite('tmpinputimg.png', cv2.hconcat([img, tmpMask]))
cv2.imwrite('tmpmaskImg.png', cv2.hconcat([empty, tmpMask]))

# cv2.imshow("resized image", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# width = int(img.shape[1] * 2)
# height = img.shape[0]
# dim = (width, height)

# resized = cv2.resize(img, dim, interpolation= cv2.)

# cv2.imshow("resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()