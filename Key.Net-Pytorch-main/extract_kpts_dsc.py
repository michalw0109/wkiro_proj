import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from model.extraction_tools import initialize_networks, compute_kpts_desc, create_result_dir
from model.config_files.keynet_configs import keynet_config
import cv2
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


def norm(image):
    return image

def scale_image(image, scale_factor=1.5):
    """Scales (resizes) an RGB image by a given factor."""
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    return np.clip(np.asarray(cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR), np.float32), 0, 1)

def add_gaussian_noise(image, mean=0, std=0.3):
    """Adds Gaussian noise to an RGB image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)

def add_salt_and_pepper_noise(image, salt_prob=0.08, pepper_prob=0.08):
    """Adds salt-and-pepper noise to an RGB image."""
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    # Salt (white pixels)
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1], :] = 1

    # Pepper (black pixels)
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_image

def add_speckle_noise(image, intensity=0.4):
    """Adds speckle noise to an RGB image."""
    noise = np.random.randn(*image.shape).astype(np.float32) * intensity
    return np.clip(image + image * noise, 0, 1)  # Multiplicative noise

def apply_color_jitter(image, brightness=0.7, contrast=2, saturation=1.5):
    """Applies color jittering (brightness, contrast, saturation) to an RGB image."""


    # Apply brightness and contrast
    image = np.clip(image * contrast + (brightness - 1.0), 0, 1)

    # Convert to HSV to modify saturation
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)  # Modify saturation


    return np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255., np.float32)


def apply_gaussian_blur(image, kernel_size=11, sigma=0):
    """Applies Gaussian blur to an RGB image."""
    return np.clip(np.asarray(cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma), np.float32), 0, 1)


def reduce_colors(image, num_colors=16):
    """Reduces the number of colors in an RGB image using k-means clustering."""
    img = image.reshape(-1, 3).astype(np.float32)  # Reshape to (N, 3)

    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Replace pixels with cluster centers
    reduced_img = centers[labels.flatten()].reshape(image.shape)
    return reduced_img

def MSE(img1, img2):
    # 1. Mean Squared Error (MSE)
    return 100000*np.mean((img1 - img2) ** 2)

def SSIM(img1, img2):
    # 2. Structural Similarity Index (SSIM)
    ssim_val, _ = ssim(img1, img2, full=True, data_range=1.0)
    return ssim_val


def PSNR(img1, img2):
    # 3. Peak Signal-to-Noise Ratio (PSNR)
    return cv2.PSNR(img1, img2)

def COSSIM(img1, img2):
    # 4. Cosine Similarity (Flattened version)
    flat_img1 = img1.flatten()
    flat_img2 = img2.flatten()
    return np.dot(flat_img1, flat_img2) / (np.linalg.norm(flat_img1) * np.linalg.norm(flat_img2))

def get_metrics(img1_path, img2_path):
    img1 = np.asarray(cv2.imread(img1_path, 0) / 255., np.float32)
    img2 = np.asarray(cv2.imread(img2_path, 0) / 255., np.float32)
    return MSE(img1, img2), SSIM(img1, img2), PSNR(img1, img2), COSSIM(img1, img2)

def gen_features(image, keynet_model, desc_model, conf, device, num_kpts, result_path, image_suf, f, scaled, scaling):

    im_gray = np.clip(np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.float32), 0, 1)  #

    xys, desc = compute_kpts_desc(im_gray, keynet_model, desc_model, conf, device, num_points=num_kpts)

    if(scaled):
        image = scale_image(image, scaling)
        xys[:, :2] *= scaling


    np.save(result_path + image_suf + '.kpt', xys)
    np.save(result_path + image_suf + '.dsc', desc)
    show_results(result_path + image_suf, image)

    mse_val, psnr_val, ssim_val, cos_sim = get_metrics(result_path + "_keypoints.jpg",
                                                       result_path + image_suf + "keypoints.jpg")
    f.write(f"{image_suf},{mse_val:.4f},{psnr_val:.4f},{ssim_val:.4f},{cos_sim:.4f}\n")


def extract_features():
    now = datetime.now()

    # Format date and time as dd_mm_yyyy_hh_mm_ss
    formatted_date_time = now.strftime("%d_%m_%Y_%H_%M_%S")

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Key.Net PyTorch + HyNet local descriptor.'
                                                 'It returns local features as:'
                                                 'kpts: Num_kpts x 4 - [x, y, scale, score]'
                                                 'desc: Num_kpts x 128')

    parser.add_argument('--list-images', type=str, help='File containing the image paths for extracting features.',
                        required=True)

    parser.add_argument('--root-images', type=str, default='',
                        help='Indicates the root of the directory containing the images.'
                       'The code will copy the structure and save the extracted features accordingly.')

    parser.add_argument('--method-name', type=str, default='keynet_hynet_default',
                        help='The output name of the method.')

    parser.add_argument('--results-dir', type=str, default='extracted_features/',
                        help='The output path to save the extracted keypoint.')

    parser.add_argument('--config-file', type=str, default='KeyNet_default_config',
                        help='Indicates the configuration file to load Key.Net.')

    parser.add_argument('--num-kpts', type=int, default=5000,
                        help='Indicates the maximum number of keypoints to be extracted.')

    parser.add_argument('--gpu-visible-devices', type=str, default='0',
                        help='Indicates the device where model should run')


    args = parser.parse_known_args()[0]

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Read Key.Net model and extraction configuration
    conf = keynet_config[args.config_file]
    keynet_model, desc_model = initialize_networks(conf)

    # read image and extract keypoints and descriptors
    f = open(args.list_images, "r")
    # for path_to_image in f:
    lines = f.readlines()

    for idx_im in tqdm(range(len(lines))):
        image_name = lines[idx_im].split('\n')[0]
        im_path = os.path.join(args.root_images, image_name+".jpg")

        result_path = os.path.join(args.results_dir, formatted_date_time, image_name, image_name)
        create_result_dir(result_path)

        with open(os.path.join(args.results_dir, formatted_date_time, image_name + "results.txt"), "w") as f:

            f.write("img,MSE,PSNR,SSIM,CosSimilarity\n")

            im_np = np.asarray(cv2.imread(im_path) / 255., np.float32)
            gen_features(im_np, keynet_model, desc_model, conf, device, args.num_kpts, result_path, "_", f, False, 0.5)


            im_scaled2_5 = scale_image(im_np, 2.5)#0.4
            im_scaled2 = scale_image(im_np, 2)#0.5
            im_scaled1_6 = scale_image(im_np, 1.6)#0.625
            im_scaled1_25 = scale_image(im_np, 1.25)#0.8
            im_scaled0_8 = scale_image(im_np, 0.8)#1.25
            im_scaled0_625 = scale_image(im_np, 0.625)#1.6
            im_scaled0_5 = scale_image(im_np, 0.5)#2
            im_scaled0_4 = scale_image(im_np, 0.4)#2.5
            im_scaled0_25 = scale_image(im_np, 0.25)#4
            im_scaled0_2 = scale_image(im_np, 0.2)#5
            im_scaled2_5_suf = "im_scaled_2.5"
            im_scaled2_suf = "im_scaled_2"
            im_scaled1_6_suf = "im_scaled_1.6"
            im_scaled1_25_suf = "im_scaled_1.25"
            im_scaled0_8_suf = "im_scaled_0.8"
            im_scaled0_625_suf = "im_scaled_0.625"
            im_scaled0_5_suf = "im_scaled_0.5"
            im_scaled0_4_suf = "im_scaled_0.4"
            im_scaled0_25_suf = "im_scaled_0.25"
            im_scaled0_2_suf = "im_scaled_0.2"
            gen_features(im_scaled2_5, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled2_5_suf, f, True, 0.4)
            gen_features(im_scaled2, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled2_suf, f, True, 0.5)
            gen_features(im_scaled1_6, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled1_6_suf, f, True, 0.625)
            gen_features(im_scaled1_25, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled1_25_suf, f, True, 0.8)
            gen_features(im_scaled0_8, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_8_suf, f, True, 1.25)
            gen_features(im_scaled0_625, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_625_suf, f, True, 1.6)
            gen_features(im_scaled0_5, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_5_suf, f, True, 2)
            gen_features(im_scaled0_4, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_4_suf, f, True, 2.5)
            gen_features(im_scaled0_25, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_25_suf, f, True, 4)
            gen_features(im_scaled0_2, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_scaled0_2_suf, f, True, 5)



            im_gnoise0_1 = add_gaussian_noise(im_np, 0,0.1)
            im_gnoise0_3 = add_gaussian_noise(im_np, 0,0.3)
            im_gnoise0_5 = add_gaussian_noise(im_np, 0,0.5)
            im_gnoise0_7 = add_gaussian_noise(im_np, 0,0.7)
            im_gnoise0_9 = add_gaussian_noise(im_np, 0,0.9)
            im_gnoise1_1 = add_gaussian_noise(im_np, 0,1.1)
            im_gnoise0_1_suf = "GaussNoise_std_0.1"
            im_gnoise0_3_suf = "GaussNoise_std_0.3"
            im_gnoise0_5_suf = "GaussNoise_std_0.5"
            im_gnoise0_7_suf = "GaussNoise_std_0.7"
            im_gnoise0_9_suf = "GaussNoise_std_0.9"
            im_gnoise1_1_suf = "GaussNoise_std_1.1"
            gen_features(im_gnoise0_1, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise0_1_suf, f, False, 1)
            gen_features(im_gnoise0_3, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise0_3_suf, f, False, 1)
            gen_features(im_gnoise0_5, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise0_5_suf, f, False, 1)
            gen_features(im_gnoise0_7, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise0_7_suf, f, False, 1)
            gen_features(im_gnoise0_9, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise0_9_suf, f, False, 1)
            gen_features(im_gnoise1_1, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gnoise1_1_suf, f, False, 1)


            im_spnoise0_02 = add_salt_and_pepper_noise(im_np, 0.02, 0.02)
            im_spnoise0_04 = add_salt_and_pepper_noise(im_np, 0.04, 0.04)
            im_spnoise0_06 = add_salt_and_pepper_noise(im_np, 0.06, 0.06)
            im_spnoise0_08 = add_salt_and_pepper_noise(im_np, 0.08, 0.08)
            im_spnoise0_10 = add_salt_and_pepper_noise(im_np, 0.10, 0.10)
            im_spnoise0_12 = add_salt_and_pepper_noise(im_np, 0.12, 0.12)
            im_spnoise0_02_suf = "SaltPepperNoise_0.02"
            im_spnoise0_04_suf = "SaltPepperNoise_0.04"
            im_spnoise0_06_suf = "SaltPepperNoise_0.06"
            im_spnoise0_08_suf = "SaltPepperNoise_0.08"
            im_spnoise0_10_suf = "SaltPepperNoise_0.10"
            im_spnoise0_12_suf = "SaltPepperNoise_0.12"
            gen_features(im_spnoise0_02, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_02_suf, f, False, 1)
            gen_features(im_spnoise0_04, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_04_suf, f, False, 1)
            gen_features(im_spnoise0_06, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_06_suf, f, False, 1)
            gen_features(im_spnoise0_08, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_08_suf, f, False, 1)
            gen_features(im_spnoise0_10, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_10_suf, f, False, 1)
            gen_features(im_spnoise0_12, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_spnoise0_12_suf, f, False, 1)


            im_snoise0_2 = add_speckle_noise(im_np)
            im_snoise0_3 = add_speckle_noise(im_np)
            im_snoise0_4 = add_speckle_noise(im_np)
            im_snoise0_5 = add_speckle_noise(im_np)
            im_snoise0_6 = add_speckle_noise(im_np)
            im_snoise0_7 = add_speckle_noise(im_np)
            im_snoise0_2_suf = "SpeckleNoise_0.2"
            im_snoise0_3_suf = "SpeckleNoise_0.3"
            im_snoise0_4_suf = "SpeckleNoise_0.4"
            im_snoise0_5_suf = "SpeckleNoise_0.5"
            im_snoise0_6_suf = "SpeckleNoise_0.6"
            im_snoise0_7_suf = "SpeckleNoise_0.7"
            gen_features(im_snoise0_2, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_2_suf, f, False, 1)
            gen_features(im_snoise0_3, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_3_suf, f, False, 1)
            gen_features(im_snoise0_4, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_4_suf, f, False, 1)
            gen_features(im_snoise0_5, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_5_suf, f, False, 1)
            gen_features(im_snoise0_6, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_6_suf, f, False, 1)
            gen_features(im_snoise0_7, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_snoise0_7_suf, f, False, 1)

            #im_jitter = apply_color_jitter(im_np, 0.5, 0.5, 0.5)
            #im_jitter_suf = "ColorJitter"
            #gen_features(im_jitter, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_jitter_suf, f, False, 0.5)

            im_gblur5 = apply_gaussian_blur(im_np, 5)
            im_gblur7 = apply_gaussian_blur(im_np, 7)
            im_gblur9 = apply_gaussian_blur(im_np, 9)
            im_gblur11 = apply_gaussian_blur(im_np, 11)
            im_gblur13 = apply_gaussian_blur(im_np, 13)
            im_gblur15 = apply_gaussian_blur(im_np, 15)
            im_gblur5_suf = "GaussBlur_5"
            im_gblur7_suf = "GaussBlur_7"
            im_gblur9_suf = "GaussBlur_9"
            im_gblur11_suf = "GaussBlur_11"
            im_gblur13_suf = "GaussBlur_13"
            im_gblur15_suf = "GaussBlur_15"
            gen_features(im_gblur5, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur5_suf, f, False, 0.5)
            gen_features(im_gblur7, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur7_suf, f, False, 0.5)
            gen_features(im_gblur9, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur9_suf, f, False, 0.5)
            gen_features(im_gblur11, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur11_suf, f, False, 0.5)
            gen_features(im_gblur13, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur13_suf, f, False, 0.5)
            gen_features(im_gblur15, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_gblur15_suf, f, False, 0.5)

            im_reduce32 = reduce_colors(im_np, 32)
            im_reduce16 = reduce_colors(im_np, 16)
            im_reduce8 = reduce_colors(im_np, 8)
            im_reduce4 = reduce_colors(im_np, 4)
            im_reduce32_suf = "ColorReduce_32"
            im_reduce16_suf = "ColorReduce_16"
            im_reduce8_suf = "ColorReduce_8"
            im_reduce4_suf = "ColorReduce_4"
            gen_features(im_reduce32, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_reduce32_suf, f, False, 0.5)
            gen_features(im_reduce16, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_reduce16_suf, f, False, 0.5)
            gen_features(im_reduce8, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_reduce8_suf, f, False, 0.5)
            gen_features(im_reduce4, keynet_model, desc_model, conf, device, args.num_kpts, result_path, im_reduce4_suf, f, False, 0.5)





#moje

def show_results(result_path, image):


    #keypoints_path = os.path.join(args.results_dir, formatted_date_time, (image_name + ".kpt.npy"))


    # Load image in color mode

    # Load keypoints
    keypoints = np.load(result_path+ ".kpt.npy")  # Shape should be (N, 4) -> [x, y, scale, score]
    canvas_height, canvas_width, canvas_depth = image.shape

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    cv2.imwrite(result_path+".jpg", image*255.)

    # Draw keypoints on the image
    for kp in keypoints:
        x, y, scale, score = kp  # Extract keypoint details
        radius = int(scale*2)  # Use scale to define circle size
        color = (0, 0, 255)  # Green color in BGR format
        thickness = 1  # Circle thickness

        # Draw circle at (x, y) with radius proportional to scale
        cv2.circle(image, (int(x), int(y)), radius, color, thickness)

        temp_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        cv2.circle(temp_canvas, (int(x), int(y)), radius, (20), -1)  # 200 is the intensity of the grey color


        # Add the circle to the main canvas (accumulate intensity)
        canvas = cv2.add(canvas, temp_canvas)

    canvas = np.clip(canvas, 0, 255)

    # Display the image with keypoints
    # cv2.imshow("Keypoints", image)
    # cv2.waitKey(0)  # Wait for key press
    # cv2.imshow("Keypoints", canvas)
    # cv2.waitKey(0)  # Wait for key press
    #
    # cv2.destroyAllWindows()  # Close the window

    # Optionally, save the image
    cv2.imwrite(result_path + "with_keypoints.jpg", image*255.)
    cv2.imwrite(result_path + "keypoints.jpg", canvas)



if __name__ == '__main__':
    extract_features()
    # extract_features(scale_image)
    # extract_features(add_gaussian_noise)
    # extract_features(add_salt_and_pepper_noise)
    # extract_features(add_speckle_noise)
    # extract_features(apply_color_jitter)
    # extract_features(apply_gaussian_blur)
    # extract_features(reduce_colors)
    #show_results("1.jpg", "03_04_2025_11_59_02")
