import numpy as np
import scipy
import cv2


def gabor_filter(im, orient, freq, kx=0.65, ky=0.65):
    """
    Gabor filter is a linear filter used for edge detection. Gabor filter can be viewed as a sinusoidal plane of
    particular frequency and orientation, modulated by a Gaussian envelope.
    :param im: asdasdasdas
    :param orient:
    :param freq:
    :param kx:
    :param ky:
    :return: imagem com filtro de gabor aplicado,limites das coordenadas
    """
    angle_inc = 3
    im = np.double(im)
    rows, cols = im.shape
    return_img = np.zeros((rows, cols))

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq_1d = freq.flatten()
    frequency_ind = np.array(np.where(freq_1d > 0))
    non_zero_elems_in_freq = freq_1d[frequency_ind]
    unfreq = np.unique(non_zero_elems_in_freq)
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
    unfreq = np.unique(freq_1d)
    unfreq = np.unique(frequency_ind)
    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angle_inc' increments.
    sigma_x = 1 / unfreq * kx
    sigma_y = 1 / unfreq * ky
    block_size = int(np.round(3 * np.max([sigma_x, sigma_y])))
    array = np.linspace(-block_size, block_size, (2 * block_size + 1))
    x, y = np.meshgrid(array, array)

    # gabor filter equation
    reffilter = np.exp(-(((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y)))) * \
                np.cos(2 * np.pi * unfreq[0] * x)
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180 // angle_inc, filt_rows, filt_cols)))

    # Generate rotated versions of the filter.
    for degree in range(0, 180 // angle_inc):
        rot_filt = scipy.ndimage.rotate(reffilter, -(degree * angle_inc + 90), reshape=False)
        gabor_filter[degree] = rot_filt

    # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angle_inc)
    maxorientindex = np.round(180 / angle_inc)
    orientindex = np.round(orient / np.pi * 180 / angle_inc)
    for i in range(0, rows // 16):
        for j in range(0, cols // 16):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    # Find indices of matrix points greater than maxsze from the image boundary
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq > 0)
    finalind = \
        np.where((valid_row > block_size) & (valid_row < rows - block_size) & (valid_col > block_size) & (
                    valid_col < cols - block_size))

    limite_colun = []

    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]];
        c = valid_col[finalind[0][k]]
        if block_size < c < (cols - block_size):
            limite_colun.append(c)
        img_block = im[r - block_size:r + block_size + 1][:, c - block_size:c + block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1])

    linha_ini, linha_fin = valid_row[finalind[0][0]], valid_row[finalind[0][np.shape(finalind)[1] - 1]]

    limite_linha = [linha_ini, linha_fin]
    limite_colun = [min(limite_colun), max(limite_colun)]

    gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

    return gabor_img, limite_linha, limite_colun
