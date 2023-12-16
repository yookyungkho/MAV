import os
import argparse
import numpy as np

def main(args):
    print(f"\n------------------------------------------------------")
    print(f"Calculate Total results for {args.result_dir}")
    print(f"------------------------------------------------------\n")
    
    seed_list = [13, 21, 42, 87, 100]

    file_name = args.result_dir.split("/")[1]
    result_file = f"{args.result_dir}/total_results.txt"
    

    accs = []

    with open(result_file, "w") as writer:
        writer.write(f"*** Result for {args.result_dir} ***\n")
        writer.write("\n")
        
        for seed in seed_list:
            # Collect all acc values of 5 seeds
            f = open(f"{args.result_dir}/seed{seed}/test_results_{args.task}.txt")
            lines = f.readlines()
            test_acc = float(lines[1].split("= ")[1].rstrip("\n"))*100
            accs.append(test_acc)
            writer.write(f"seed {seed} test acc: {test_acc: .2f}   <-{test_acc}\n")
            print(f"seed {seed} test acc: {test_acc: .2f}   <-{test_acc}\n")

            if args.bin_mode == "remove":
                rm_pth = f"{args.result_dir}/seed{seed}/pytorch_model.bin"
                if os.path.isfile(rm_pth):
                    os.remove(rm_pth)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        
        writer.write(f"\n=> mean: {mean_acc: .2f}, std: {std_acc: .3f}\n")
        print(f"\n=> mean: {mean_acc: .2f}, std: {std_acc: .3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result_argparse')

    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--bin_mode", type=str, default="save")

    result_args = parser.parse_args()

    main(result_args)
