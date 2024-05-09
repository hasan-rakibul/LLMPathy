import os
import time
import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import tqdm

import omegaconf

from preprocess import SSLDataModule, DataModule
from model import SSLRoberta
from utils import seed_everything

def main(
    w_ulb = 10,
    samp_ssl = 5
):
    pd.options.mode.copy_on_write = True # to avoid SettingWithCopyWarning; details: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    config_file = "config/config.yaml"
    config = omegaconf.OmegaConf.load(config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm_ssl = SSLDataModule(config)
    train_dl_labelled, train_dl_unlabelled, y_mean, y_std = dm_ssl.get_dataloader(
        data_file=config.data.train_file,
        send_label=True,
        shuffle=False
    )

    dm_val = DataModule(config)
    val_dl = dm_val.get_dataloader(
        data_file=config.data.val_file,
        send_label=True,
        shuffle=False
    )

    model = SSLRoberta(config)
    model_1 = SSLRoberta(config)

    model = model.to(device)
    model_1 = model_1.to(device)

    # Seed RNGs
    seed_everything(config.seed)

    output = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output, exist_ok=True)

    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr, 
                             weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config.train.lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=config.train.lr, 
                               weight_decay=config.train.weight_decay)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, config.train.lr_step_period)

    with open(os.path.join(output, "log.csv"), "a") as f:

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)
            optim_1.load_state_dict(checkpoint['opt_dict_1'])
            scheduler_1.load_state_dict(checkpoint['scheduler_dict_1'])

            np_rndstate_chkpt = checkpoint['np_rndstate']
            trch_rndstate_chkpt = checkpoint['trch_rndstate']

            np.random.set_state(np_rndstate_chkpt)
            torch.set_rng_state(trch_rndstate_chkpt)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, config.train.num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:

                start_time = time.time()

                # if device.type == "cuda":
                #     for i in range(torch.cuda.device_count()):
                #         torch.cuda.reset_peak_memory_stats(i)
                
                if phase == "train":
                    loss_tr, loss_reg_0, loss_reg_1, cps, cps_l, cps_s,\
                    yhat_0, yhat_1, y, mean_0_ls, mean_1_ls, var_0_ls, var_1_ls \
                        = run_epoch(
                            model, 
                            model_1, 
                            train_dl_labelled, 
                            train_dl_unlabelled, 
                            phase == "train", # means train = True
                            optim, 
                            optim_1,
                            device=device,
                            w_ulb=w_ulb, 
                            y_mean=y_mean, 
                            y_std=y_std,
                            samp_ssl=samp_ssl
                        )

                    pcc_0 = pearsonr(y, yhat_0)[0] # [0] is the correlation coefficient statistic, [1] is the p-value
                    pcc_1 = pearsonr(y, yhat_1)[0]

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss_tr,
                                                                pcc_0,
                                                                pcc_1,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                config.train.batch_size,
                                                                loss_reg_0,
                                                                cps))
                    f.flush()
                
                    with open(os.path.join(output, "train_pred_{}.csv".format(epoch)), "w") as f_trnpred:
                        for clmn in range(mean_0_ls.shape[1]):
                            f_trnpred.write("m_0_{},".format(clmn))
                        for clmn in range(mean_1_ls.shape[1]):
                            f_trnpred.write("m_1_{},".format(clmn))
                        for clmn in range(var_0_ls.shape[1]):
                            f_trnpred.write("v_0_{},".format(clmn))
                        for clmn in range(var_1_ls.shape[1]):
                            f_trnpred.write("v_1_{},".format(clmn))
                        f_trnpred.write("\n".format(clmn))
                        
                        for rw in range(mean_0_ls.shape[0]):
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                            for clmn in range(mean_1_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_1_ls[rw, clmn]))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                            for clmn in range(var_1_ls.shape[1]):
                                f_trnpred.write("{},".format(var_1_ls[rw, clmn]))
                            f_trnpred.write("\n".format(clmn))

                
                else:
                    loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(
                        model = model, 
                        model_1 = model_1, 
                        dataloader = val_dl, 
                        train = False, 
                        optim = None,
                        device=device,
                        y_mean = y_mean, 
                        y_std = y_std, 
                        samp_ssl = samp_ssl
                    )

                    pcc = pearsonr(y, yhat)[0]
                    loss = loss_valit

                    with open(os.path.join(output, "z_{}_epch{}_prd.csv".format(phase, epoch)), "a") as pred_out:
                        pred_out.write("yhat,y,var_hat, var_e, var_a\n")
                        for pred_itr in range(y.shape[0]):
                            pred_out.write("{},{},{},{},{}\n".format(yhat[pred_itr],
                            y[pred_itr], 
                            var_hat[pred_itr], 
                            var_e[pred_itr], 
                            var_a[pred_itr]))
                        pred_out.flush()
                    
                    with open(os.path.join(output, "val_predmcd0_{}.csv".format(epoch)), "w") as f_trnpred:
                        for clmn in range(mean_0_ls.shape[1]):
                            f_trnpred.write("m_0_{},".format(clmn))
                        for clmn in range(var_0_ls.shape[1]):
                            f_trnpred.write("v_0_{},".format(clmn))
                        f_trnpred.write("\n".format(clmn))
                        
                        for rw in range(mean_0_ls.shape[0]):
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                            f_trnpred.write("\n".format(clmn))


                    f.write("{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                pcc,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                config.train.batch_size,
                                                                0,
                                                                0))
            
                    

                    f.write("\n")
                    f.flush()


            
            scheduler.step()
            scheduler_1.step()

            best_model_loss = loss_valit

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_1': model_1.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                "best_model_loss": best_model_loss,
                'pcc': pcc,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'opt_dict_1': optim_1.state_dict(),
                'scheduler_dict_1': scheduler_1.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            
            if best_model_loss < bestLoss:
                print("saved best because {} < {}".format(best_model_loss, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = best_model_loss


        # Load best weights
        if config.train.num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)

            f.write("Best validation loss {} from epoch {}, PCC {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["pcc"]))
            f.flush()


def run_epoch(model, 
            model_1, 
            dataloader_lb, 
            dataloader_unlb_0, 
            train, 
            optim, 
            optim_1,
            device,
            w_ulb,  
            y_mean, 
            y_std,
            samp_ssl):
    
    model.train(train)
    model_1.train(train)

    total = 0  
    total_reg = 0 
    total_reg_1 = 0

    total_cps = 0
    total_cps_0 = 0
    total_cps_1 = 0

    n = 0 

    yhat_0 = []
    yhat_1 = []
    y = []

    mean2s_0_stack_ls = []
    mean2s_1_stack_ls = []
    var1s_0_stack_ls = []
    var1s_1_stack_ls = []

    torch.set_grad_enabled(train)

    for train_iter, (batch_lb, batch_unlb_0) in enumerate(zip(dataloader_lb, dataloader_unlb_0)):

        input_ids = batch_unlb_0["input_ids"].to(device)
        attention_mask = batch_unlb_0["attention_mask"].to(device)

        all_output_unlb_0_pred_0, var_unlb_0_pred_0 = model(
            input_ids,
            attention_mask
        )
        all_output_unlb_1_pred_0, var_unlb_1_pred_0 = model_1(
            input_ids,
            attention_mask
        )
        
        mean1s_0 = []
        mean2s_0 = []
        var1s_0 = []

        mean1s_1 = []
        mean2s_1 = []
        var1s_1 = []

        with torch.no_grad():
            for _ in range(samp_ssl):
                mean1_raw_0, var1_raw_0 = model(
                    input_ids,
                    attention_mask
                )
                mean1_0 = mean1_raw_0.view(-1)
                var1_0 = var1_raw_0.view(-1)

                mean1s_0.append(mean1_0** 2)
                mean2s_0.append(mean1_0)
                var1s_0.append(var1_0)

                mean1_raw_1, var1_raw_1 = model_1(
                    input_ids,
                    attention_mask
                )
                mean1_1 = mean1_raw_1.view(-1)
                var1_1 = var1_raw_1.view(-1)

                mean1s_1.append(mean1_1** 2)
                mean2s_1.append(mean1_1)
                var1s_1.append(var1_1)


        mean2s_0_stack = torch.stack(mean2s_0, dim=1).to("cpu").detach().numpy()
        mean2s_0_stack_ls.append(mean2s_0_stack)
        var1s_0_stack = torch.stack(var1s_0, dim=1).to("cpu").detach().numpy()
        var1s_0_stack_ls.append(var1s_0_stack)

        mean1s_0_ = torch.stack(mean1s_0, dim=0).mean(dim=0)
        mean2s_0_ = torch.stack(mean2s_0, dim=0).mean(dim=0)
        var1s_0_ = torch.stack(var1s_0, dim=0).mean(dim=0)

        mean2s_1_stack = torch.stack(mean2s_1, dim=1).to("cpu").detach().numpy()
        mean2s_1_stack_ls.append(mean2s_1_stack)
        var1s_1_stack = torch.stack(var1s_1, dim=1).to("cpu").detach().numpy()
        var1s_1_stack_ls.append(var1s_1_stack)

        mean1s_1_ = torch.stack(mean1s_1, dim=0).mean(dim=0)
        mean2s_1_ = torch.stack(mean2s_1, dim=0).mean(dim=0)
        var1s_1_ = torch.stack(var1s_1, dim=0).mean(dim=0)


        all_output_unlb_0_pslb = mean2s_0_
        all_output_unlb_1_pslb = mean2s_1_

        avg_mean01 = (all_output_unlb_0_pslb + all_output_unlb_1_pslb)/2
        avg_var01 = (var1s_0_ + var1s_1_)/2

        loss_mse_cps_0 = ((all_output_unlb_0_pred_0.view(-1) - avg_mean01)**2)
        loss_mse_cps_1 = ((all_output_unlb_1_pred_0.view(-1) - avg_mean01)**2)

        loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_0) + avg_var01 )
        loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_1) + avg_var01 )

        loss_reg_cps0 = loss_cmb_cps_0.mean()
        loss_reg_cps1 = loss_cmb_cps_1.mean()

        
        var_loss_ulb_0 = ((var_unlb_0_pred_0.view(-1) - avg_var01)**2).mean()
        var_loss_ulb_1 = ((var_unlb_1_pred_0.view(-1) - avg_var01)**2).mean()


        loss_reg_cps = (loss_reg_cps0 + loss_reg_cps1) + (var_loss_ulb_0 + var_loss_ulb_1)

        outcome = batch_lb["labels"]
        y.append(outcome.detach().cpu().numpy())

        input_ids = batch_lb["input_ids"].to(device)
        attention_mask = batch_lb["attention_mask"].to(device)

        outcome = outcome.to(device)

        all_output = model(
            input_ids,
            attention_mask
        )
        all_output_1 = model_1(
            input_ids,
            attention_mask
        )             

        mean_raw, var_raw = all_output
        mean = mean_raw.view(-1)
        var = var_raw.view(-1)

        mean_1_raw, var_1_raw = all_output_1
        mean_1 = mean_1_raw.view(-1)
        var_1 = var_1_raw.view(-1)

        loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
        loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)
        loss2 = (var + var_1) / 2
        loss = .5 * (loss1 + loss2)

        loss_reg_0 = loss.mean()
        yhat_0.append(all_output[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)

        loss_mse_1 = (mean_1 - (outcome - y_mean) / y_std) ** 2
        loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse_1)
        loss2_1 = (var + var_1) / 2
        loss_1 = .5 * (loss1_1 + loss2_1)

        loss_reg_1 = loss_1.mean()
        yhat_1.append(all_output_1[0].view(-1).to("cpu").detach().numpy() * y_std + y_mean)


        loss_reg = (loss_reg_0 + loss_reg_1)

        loss = loss_reg + w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean()

        
        if train:
            optim.zero_grad()
            optim_1.zero_grad()
            loss.backward()
            optim.step()
            optim_1.step()

        total += loss.item() * outcome.size(0)
        total_reg += loss_reg_0.item() * outcome.size(0)
        total_reg_1 += loss_reg_1.item() * outcome.size(0)

        total_cps += loss_reg_cps.item() * outcome.size(0)
        total_cps_0 += loss_reg_cps0.item() * outcome.size(0)
        total_cps_1 += loss_reg_cps1.item() * outcome.size(0)

        n += outcome.size(0)

        if train_iter % 10 == 0:
            print("phase {} itr {}/{}: ls {:.2f}({:.2f}) rg0 {:.4f} ({:.2f}) rg1 {:.4f} ({:.2f}) cps {:.4f} ({:.2f}) cps0 {:.4f} ({:.2f}) cps1 {:.4f} ({:.2f})".format(train,
                train_iter, len(dataloader_lb), 
                total / n, loss.item(), 
                total_reg/n, loss_reg_0.item(), 
                total_reg_1/n, loss_reg_1.item(), 
                total_cps/n, loss_reg_cps.item(),
                total_cps_0/n, loss_reg_cps0.item(),
                total_cps_1/n, loss_reg_cps1.item()), flush = True)


    yhat_0 = np.concatenate(yhat_0)
    yhat_1 = np.concatenate(yhat_1)
        

    y = np.concatenate(y)

    mean2s_0_stack_ls = np.concatenate(mean2s_0_stack_ls)
    mean2s_1_stack_ls = np.concatenate(mean2s_1_stack_ls)
    var1s_0_stack_ls = np.concatenate(var1s_0_stack_ls)
    var1s_1_stack_ls = np.concatenate(var1s_1_stack_ls)

    return total / n, total_reg / n, total_reg_1 / n, total_cps / n, total_cps_0 / n, total_cps_1 / n, yhat_0, yhat_1, y, mean2s_0_stack_ls, mean2s_1_stack_ls, var1s_0_stack_ls, var1s_1_stack_ls

def run_epoch_val(
        model, 
        model_1, 
        dataloader, 
        train, 
        optim,
        device,
        y_mean, 
        y_std, 
        samp_ssl
    ):


    model.train(False)
    model_1.train(False)

    total = 0 
    n = 0   

    yhat = []
    y = []

    var_hat = []
    var_e = []
    var_a = []

    mean2s_0_stack_ls = []
    var1s_0_stack_ls = []
    mean2s_0_stack_ls_m1 = []
    var1s_0_stack_ls_m1 = []

    mean2s_0_stack_ls_avg = []
    var1s_0_stack_ls_avg = []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                outcome = batch["labels"]
                y.append(outcome.numpy())
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outcome = outcome.to(device)

                mean1s = []
                mean2s = []
                var1s = []

                mean1s_m1 = []
                mean2s_m1 = []
                var1s_m1 = []

                for _ in range(samp_ssl):
                    all_ouput = model(input_ids, attention_mask)
                    mean1_raw, var1_raw = all_ouput

                    mean1 = mean1_raw.view(-1)
                    var1 = var1_raw.view(-1)

                    mean1s.append(mean1** 2)
                    mean2s.append(mean1)
                    var1s.append(torch.exp(var1))

                    all_ouput_m1 = model_1(input_ids, attention_mask)
                    mean1_raw_m1, var1_raw_m1 = all_ouput_m1

                    mean1_m1 = mean1_raw_m1.view(-1)
                    var1_m1 = var1_raw_m1.view(-1)

                    mean1s_m1.append(mean1_m1** 2)
                    mean2s_m1.append(mean1_m1)
                    var1s_m1.append(torch.exp(var1_m1))


                mean2s_0_stack = torch.stack(mean2s, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls.append(mean2s_0_stack)
                var1s_0_stack = torch.stack(var1s, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls.append(var1s_0_stack)

                mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
                mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
                var1s_ = torch.stack(var1s, dim=0).mean(dim=0)


                mean2s_0_stack_m1 = torch.stack(mean2s_m1, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls_m1.append(mean2s_0_stack_m1)
                var1s_0_stack_m1 = torch.stack(var1s_m1, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls_m1.append(var1s_0_stack_m1)

                mean2s_0_stack_ls_avg.append((mean2s_0_stack + mean2s_0_stack_m1) / 2)
                var1s_0_stack_ls_avg.append((var1s_0_stack + var1s_0_stack_m1) / 2)


                mean1s_m1_ = torch.stack(mean1s_m1, dim=0).mean(dim=0)
                mean2s_m1_ = torch.stack(mean2s_m1, dim=0).mean(dim=0)
                var1s_m1_ = torch.stack(var1s_m1, dim=0).mean(dim=0)


                var2 = mean1s_ - mean2s_ ** 2
                var_ = var1s_ + var2
                var_norm = var_ / var_.max()         

                var2_m1 = mean1s_m1_ - mean2s_m1_ ** 2
                var_m1_ = var1s_m1_ + var2_m1
                var_m1_norm = var_m1_ / var_m1_.max()

                yhat.append(((mean2s_ + mean2s_m1_) / 2).to("cpu").detach().numpy() * y_std + y_mean)
                var_hat.append(((var_norm + var_m1_norm) / 2).to("cpu").detach().numpy())
                var_e.append(((var2 + var2_m1) / 2).to("cpu").detach().numpy())
                var_a.append(((var1s_ + var1s_m1_) / 2).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss( (mean2s_ + mean2s_m1_) / 2 , (outcome - y_mean) / y_std )

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * input_ids.size(0)
                n += input_ids.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))
                pbar.update()

    yhat = np.concatenate(yhat)
    var_hat = np.concatenate(var_hat)
    var_e = np.concatenate(var_e)
    var_a = np.concatenate(var_a)
    y = np.concatenate(y)

    mean2s_0_stack_ls_avg = np.concatenate(mean2s_0_stack_ls_avg)
    var1s_0_stack_ls_avg = np.concatenate(var1s_0_stack_ls_avg)

    return total / n, yhat, y, var_hat, var_e, var_a, mean2s_0_stack_ls_avg, var1s_0_stack_ls_avg


if __name__ == '__main__':
    main()