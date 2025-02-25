
# Image Editing with Cross Attention Control

This project explores attention manipulation in text-to-image diffusion models, inspired by the paper [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/pdf/2208.01626), providing an intuitive text-based approach to image modification. By leveraging cross-attention maps in diffusion models, we can perform precise word-level image edits while maintaining the overall composition.

## Cross-Attention Visualization
Cross-attention maps help understand word localization in generated images. Below is an example of cross-attention maps for each word in a given prompt, showing how different words influence specific image regions.

![Cross Attention Example](https://github.com/user-attachments/assets/d010bb4d-d68a-4d39-b84d-128dabd13b61)

## Implementation Overview
This implementation allows controlled image editing through three primary techniques:

### 1. Saving Initial Attention Maps
- Generate an image using an initial prompt.
- Store the cross-attention maps corresponding to each token.

### 2. Word Substitution with Fixed Attention
- Replace a specific word in the prompt while keeping all other attention maps unchanged.
- Regenerate the image to observe the targeted modification without affecting the overall composition.

#### Example:
- **Original Prompt:** "a **pepperoni** pizza"
 
  ![Pepperoni Pizza](https://github.com/user-attachments/assets/8c0bdea1-f655-4596-8348-195ca8f9c7b3)

- **Modified Prompt:** "a **mushroom** pizza"
 
  ![Mushroom Pizza](https://github.com/user-attachments/assets/53aca9c9-2978-4469-bca1-869176405a86)

*Observation:* The image structure is preserved while only modifying the word-dependent features.

### 3. Attention Reweighting with Masking
- Identify the highest activation values for a target word.
- Apply a mask to selectively amplify the word’s influence while maintaining the original seed and attention maps.

#### Example:
**Prompt:** "a **fluffy** teddy bear"

- **Cross-attention weighted from -5 to 5 (No Masking):**  
  ![Download](https://github.com/user-attachments/assets/3968a2a9-d135-4e25-be68-a8046b4cabe0)

- **Masking with Low Threshold (th=0.3):**  
  ![Download](https://github.com/user-attachments/assets/976b0ae0-6028-4470-8d53-b291b4be78c8)

- **Masking with Higher Threshold (th=0.45):**  
  ![Download](https://github.com/user-attachments/assets/d9637751-d1c6-4263-8e37-50fcdb8c2181)

*Observation:* By increasing the threshold, we gain more localized control over word influence. Additional techniques like averaging attention maps across diffusion steps or compressing values before thresholding can further refine results.

A better prompt can be introduced to refine results by fixing the color:

- **New Prompt:** "a brown **fluffy** teddy bear"  
  ![Download](https://github.com/user-attachments/assets/882555b4-03e3-495d-bd9c-51c1689bed24)

-  With a different seed:
  
  ![download](https://github.com/user-attachments/assets/5edc23d0-d577-433f-aaa2-601dab2a36f0)


## Conclusion
This method provides a seamless way to edit images using only text, maintaining structural integrity while allowing for localized or global modifications. The approach enables:
- Word-specific content modifications.
- Controlled influence of text tokens using cross-attention.
- Preservation of original composition while modifying attributes.

## References
[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/pdf/2208.01626)
