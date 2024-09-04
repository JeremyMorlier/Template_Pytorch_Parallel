import os

import argparse

import json
import wandb

import time

from utils import create_dir

class Logger() :


    def __init__(self, project_name, run_name, tags = None, resume = False, id=None, args = None, mode="txt", log_dir="./") :

        self.step = 0
        self.mode = mode
        self.id = id

        args.timestamp = time.time()
        print(args)
        if mode == "wandb" :
            
            wandb_resume = None
            if resume :
                wandb_resume = "allow"
            wandb.init(
                # set the wandb project where this run will be logged
                project=project_name,
                name=run_name,
                tags=tags,
                resume=wandb_resume,
                id=id,
                # track hyperparameters and run metadata
                config=args
            )
        elif self.mode == "txt" :

            create_dir(log_dir)
            self.file = os.path.join(log_dir, project_name + "_" + run_name + ".log")
            print(self.file)
            write_header = True
            if resume :
                if os.path.isfile(self.file) :

                    # Read the first line to find the header
                    with open(self.file, "r") as file :
                        lines = file.readlines()

                        header = json.loads(lines[0])
                        if "project_name" in header and "run_name" in header :
                            print("test")
                            write_header = False

                        last_log = json.loads(lines[-1])
                        if "step" in last_log :
                            self.step = last_log["step"] + 1

            if write_header :
                with open(self.file, "w") as file :
                    header = {"project_name": project_name, "run_name": run_name, "tags": tags, "args": args.__dict__}

                    json.dump(header, file)
                    file.write("\n")

    def log(self, dictionnary) :
        # Update log with global step to better sync between modes
        dictionnary["step"] = self.step
        self.step += 1

        if self.mode == "wandb" :
            wandb.log(dictionnary)
        elif self.mode == "txt" :
            with open(self.file, "a") as file :
                json.dump(dictionnary, file)
                file.write("\n")
    def finish(self) :
        if self.mode == "wandb" :
            wandb.finish()
        
        print("Logging Finished !")

# Call the logger directly to translate a txt log to wandb
def args_parser(add_help=True) :
    parser = argparse.ArgumentParser(description="Logger Parser", add_help=add_help)

    parser.add_argument("--path", type=str, default="./default.log")

    return parser

def wandb_log(filepath) :
    if os.path.isfile(filepath) :

        with open(filepath, "r") as file :
            lines = file.readlines()

            header = json.loads(lines[0])

            # wandb init
            project_name = header["project_name"]
            run_name = header["run_name"]
            tags = header["tags"]
            args = header["args"]
            wandb.init(
                project=project_name,
                name=run_name,
                tags=tags,
                config=args
            )

            for line in lines[1:] :
                wandb.log(json.loads(line))
            wandb.finish()
    else :
        print("Log file does not exist")
if __name__ == "__main__" :

    arguments, unknown = args_parser().parse_known_args()
    if  os.path.isdir(arguments.path) :
        print("Processing Folder")
        files_list = os.listdir(arguments.path)
        for filepath in files_list :
            print("Processing: ", filepath)
            wandb_log(os.path.join(arguments.path, filepath))
    elif os.path.isfile(arguments.path) :
        print("Processing: ", arguments.path)
        wandb_log(arguments.path)

    