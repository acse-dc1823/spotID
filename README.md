# SpotID: A Leopard Individual Identifier

This project attempts to use Deep Learning to create a Leopard Individual Identifier. It encodes each leopard image into N dimensional embeddings, and then these embeddings are compared with one another through a distance metric to decide which images correspong to the same leopard not. Two methods were tried, Triplet Networks, and a modified CosFace. For more details, please read the attached paper. 

## Instructions for users:

First install the software. Two options:

1. Open terminal on computer, type on terminal:
```bash
git clone https://github.com/ese-msc-2023/irp-dc1823.git
```

2. Go to Github page for the project, click on the green <>Code button, press "Download ZIP"

Then, navigate to the software from your terminal from wherever you are in your directory:

```bash
cd path/to/irp-dc1823
```

Then, install all the required software packages for the project, typing the following on the terminal:

```bash
pip install -r requirements.txt
```

We then have all the steps to encode our images into embeddings. We navigate to "leopard_id" from irp-dc1823

```bash
cd leopard_id
```

We open "config_inference.json" with whichever text editor you have available, and edit the needed variables. If the images have not been preprocessed (not cropped), then we need to edit just two variables:

```
    ...
    "preprocess": true,
    "unprocessed_image_folder": "path/to/unprocessed_image_folders",
    ...
```

the "preprocess" flag will indicate to the software that the preprocessing pipeline needs to be ran for the images. In the preprocessing pipeline, the image will be cropped (stored in "crop_output_folder"), background removed (stored in "bg_removed_output_folder") and edge detected (stored in "base_binary_output_folder"). If the user wants any of these variables to be modified, please feel free to do so. We can now run the code to create the embeddings with the saved config file:

On the terminal, from leopard_id subdirectory (that we have navigated to above), we type:

```bash
python3 inference_embeddings.py
```

Disclaimer: The preprocessing pipeline is quite slow, approximately 10-15s per image. The model itself is very fast, it takes approximately 15s for a test dataset of 500 images. Although, the preprocessing only needs to be done once per dataset.

Now, this code saves the embeddings and the distance between them on a subdirectory. This means that we can start with the inference, checking that the matches made are actually correct or not:

Navigate to the interface folder. From leopard_id, this is:

```bash
cd ../interface
```

Run the interface:

```bash
python app.py
```

Go to your browser of choice and type:

http://127.0.0.1:5000

Now the interface will open. In it, we have the following functionality:

1. Setting a match directory. Set it with a global path (path from root, i.e. "Users/xy1234/documents/leopards"). In this directory, whenever we end the session, the images we have classified will be arranged into their individual leopards in subdirectories in this directory. There will also be a csv that outlines which image belongs to which leopard here.

2. Setting a database or opening one. This is important. This can be directly in this directory, hence just write the name (i.e. leopard_db). If this is a name never seen by the software, it will create a new db. If it is an old database, it will just open it. The benefit of this is that, we can save our progress checking the leopards, and then open the database again, and it will start the process from the last checked leopard!

3. Start comparing! The software will go over each image in the dataset. It will show, in order from most likely to least likely to be a match, the top 5 most similar images, with a confidence value displayed. 

    - The user can zoom by clicking on the image, and can also toggle between the original cropped image and the edge detected image (which will isolate the spots). If they confirm a match, then the database will link those two images together. If "no match" is clicked, no link is created and the next image is shown. If the user is not satisfied with the comparison images before the 5th comparison image is shown, "next anchor image" can be clicked.

    - The software is "intelligent", meaning that, if we link image "A" with image "B" when "A" is the "anchor", then when we get to image "B", image "A" will be skipped over the possible matches, as it would mean extra examination effort wasted. Then, don't be worried when, as you've examined a lot of leopards, fewer and fewer correct matches are shown. This is simply because all the correct matches have already been made previously!

    - When the user needs to stop or when they believe all the correct matches have been made, they can click "end session". This will create a subdirectory structure with all the matched leopards and their corresponding images, and a csv with the filepaths and the leopards they correspond to. Again, if the user needs to continue later on, they just need to write the same database name, and the existing database will be loaded!


Finally, if new images are added to the raw data ("unprocessed_image_folder" above), don't fret, the code has been adapted so that it only runs for the new images each time, so it doesn't take forever. So please don't worry, you can run it with that dataset, and it will only process the new images.


