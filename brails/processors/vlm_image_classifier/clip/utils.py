import torch, torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm

'''
Batch Data Preprocessing of models
'''
def preprocess_batch_img(batch_idx, batch_size, img_paths, aug, device):
    batch_paths = img_paths[batch_size*batch_idx:batch_size*(batch_idx+1)]
    img_list = []
    for i, p in enumerate(batch_paths):
        image = Image.open(p)
        img_list.append(image)
    image_input = torch.cat([aug(img).unsqueeze(0).to(device) for img in img_list])
    return image_input

'''
Main function to perform classficiation of an image-text model (e.g. CLIP)
Input: 
- model: model to perform classification
- text input: text prompts to perform classification and matched with image inputs to yield similarity
- img_paths: absolute paths of images to be predicted
- batch_size: (hyperparameter) number of images processed per forward iteration
- preprocess: data preprocessing pipeline before model input
- agg: aggregation method of similarity values when providing multiple prompts per class
- num_classes: number of target, candidate classes for prediction
Output: Dataframe containing model predictions, indexed by image name
'''
def predict_wrapper(model, text_input, img_paths, preprocess, device, batch_size = 200, agg = "max", num_classes = 3):
    prediction_df = pd.DataFrame()
    gap = len(text_input) // num_classes

    num_batches = len(img_paths) // batch_size
    num_batches += (len(img_paths) % batch_size != 0)
    for batch_idx in tqdm(range(num_batches)):
        image_input = preprocess_batch_img(batch_idx, batch_size, img_paths, preprocess, device)
        img_names = [path.split('/')[-1] for path in img_paths[batch_idx*batch_size:(batch_idx+1)*batch_size]]

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

        similarity = compute_similarity(image_features, text_features)
        predictions, max_sim_list = aggregate_predictions(similarity, agg_method = agg, gap = gap)
        max_sim_list = round(max_sim_list, decimals=2) #truncate digits for better clarity
        batch_df = pd.DataFrame(data = {'img_name': img_names, 'predictions':predictions, 'prob_list': max_sim_list.tolist()})
        prediction_df = pd.concat([prediction_df, batch_df], axis = 0)
    return prediction_df

'''
Input: Image feature, text feature
Output: Cosine-similarity of each text feature with given image, normalized across classes as probability values

E.g. image_feature = [im1, im2], text_feature = [t1, t2, t3]
Output: (im1) -> [[0.1, 0.7, 0.2], (im2) -> [0.3, 0.5, 0.2]]
'''
def compute_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    similarity = similarity.cpu()
    return similarity


'''
Given k text options per class(c classes), we output the probability of predicting each class by selecting the max/mean of the k options
Input: Similarity_arr, where each data point has k * c probability values
Output: Prediction, where each data point output 1 probability value per class (see example below)

E.g. similarity_arr = [0.2, 0.1, 0.15, 0.35, 0.1, 0.1] (class1 = [0.2, 0.1, 0.15], class2 = [0.35, 0.1, 0.1])
agg = max: predictions = [0.2, 0.35]
agg = mean: predictions = [0.15, 0.1833]
'''
def aggregate_predictions(similarity, agg_method, gap):
    batch_num = similarity.shape[0]
    if(agg_method == "max"):
        sim_per_class = torch.max(similarity.reshape(batch_num, -1, gap), dim = -1).values.cpu() # n x total_options -> n x 3 x options_per_class -> n x 3 (take the max probabiltiy among text prompts for each class)
    elif(agg_method == "mean"):
        sim_per_class = torch.mean(similarity.reshape(batch_num, -1, gap), dim = -1).cpu() # n x total_options -> n x 3 x options_per_class -> n x 3 (take the mean probability among text prompts for each class)
    else:
        raise Exception("unsupported aggregated method among text options")
    predictions = torch.argmax(sim_per_class, dim = -1).numpy() # n x 3 -> n
    #if(batch_idx == 0): print(sim_per_class.shape, predictions.shape)
    #predictions = predictions + 1 #convert idx to floor_num
    return predictions, sim_per_class

def round(similarity, decimals):
    scale = 10**decimals
    return (similarity*scale + 5).to(torch.int) / scale


#TODO: gt_labels != text_prompt for multi text_prompt mode
'''
Helper function that converts model's index-based predictions to user-prompted labels
'''
def pred_idx_to_labels(gt_labels, prediction_df):
    convert_map = {idx:label for idx, label in enumerate(gt_labels)}
    prediction_df['predictions'] = prediction_df['predictions'].replace(convert_map)
    return prediction_df
