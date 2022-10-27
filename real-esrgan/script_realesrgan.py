import os

def runScript(inputDirectory, outputDirectory, model="realesr-animevideov3", scale=2, format="png"):
    
    possible_models = ["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"]
    if(not model in possible_models):
        print("The model " + model + " is not available.")
        return

    possible_formats = ["jpg", "png", "webp"]
    if(not format in possible_formats):
        print("The format " + format + " is not available.")
        return

    possible_scales = [2, 3, 4]
    if(not scale in possible_scales):
        print("The scale ratio " + str(scale) + " is not available.")
        return
    
    if(not os.path.isdir(inputDirectory)):
        print("The directory " + inputDirectory + " does not exist") 
        return

    if(not os.path.isdir(outputDirectory)):
        print("The directory " + outputDirectory + " does not exist") 
        return
    
    os.system("./realesrgan-ncnn-vulkan -i " + inputDirectory + " -o " + outputDirectory + " -n " + model + " -s " + str(scale) + " -f " + format)

