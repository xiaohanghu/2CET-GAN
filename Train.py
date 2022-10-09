"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

from DataLoader import create_sample_getter
from Model import *
from Utils import *


def adv_loss(logits, target):
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    """
    zero-centered gradient penalty for real images.
    It can force the output not to change too much if the inputs are similar
    """
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    # mean(sum(g^2))
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def encode(encoder, x_z, config):
    return config.lambda_c * encoder(x_z)


def loss_D_NEN(model, config, sample):
    """
    D loss of cycle 1: N to E to N
    """
    x_n, x_e = sample.x_n.to(config.device), sample.x_e.to(config.device)

    x_e.requires_grad_()
    y_e = torch.ones(x_e.shape[0], dtype=torch.long).to(config.device)
    out_e = model.discriminator(x_e, y_e)
    # loss_adv_e_real
    loss_real_e = adv_loss(out_e, 1)
    loss_reg_e = r1_reg(out_e, x_e)

    # with fake images
    with torch.no_grad():
        c_e = generate_rand_code(x_n.size(0), config)
        x_fake_e = model.generator(x_n, c_e)

    out_fake_e = model.discriminator(x_fake_e, y_e)
    # loss_adv_e_fake
    loss_fake_e = adv_loss(out_fake_e, 0)
    loss_reg_e = config.lambda_reg * loss_reg_e
    loss = loss_real_e + loss_fake_e + loss_reg_e

    return loss, Munch(
        # total=loss.item(),
        # real_n=loss_real_n.item(),
        real_e=loss_real_e.item(),
        # fake_n=loss_fake_n.item(),
        fake_e=loss_fake_e.item(),
        # reg_n=loss_reg_n.item(),
        reg_e=loss_reg_e.item(),
    )


def loss_D_ENE(model, config, sample):
    """
    D loss of cycle 2: E to N to E
    """
    x_n, x_e = sample.x_n.to(config.device), sample.x_e.to(config.device)

    y_n = torch.zeros(x_n.shape[0], dtype=torch.long).to(config.device)

    x_n.requires_grad_()
    out_n = model.discriminator(x_n, y_n)
    # loss_adv_n_real
    loss_real_n = adv_loss(out_n, 1)
    loss_reg_n = r1_reg(out_n, x_n)

    # fake image
    with torch.no_grad():
        c_n_real = get_neutral_code(x_n.shape[0], x_n.dtype, config)
        x_fake_n = model.generator(x_e, c_n_real)

    out_fake_n = model.discriminator(x_fake_n, y_n)
    # loss_adv_e_fake
    loss_fake_n = adv_loss(out_fake_n, 0)
    loss_reg_n = config.lambda_reg * loss_reg_n
    loss = loss_real_n + loss_fake_n + loss_reg_n

    return loss, Munch(
        # total=loss.item(),
        real_n=loss_real_n.item(),
        # real_e=loss_real_e.item(),
        fake_n=loss_fake_n.item(),
        # fake_e=loss_fake_e.item(),
        reg_n=loss_reg_n.item(),
        # reg_e=loss_reg_e.item(),
    )


def loss_GE_NEN(model, config, sample):
    """
    G and E loss of cycle 1: N to E to N
    """
    x_n = sample.x_n.to(config.device)

    # n
    y_e = torch.ones(x_n.shape[0], dtype=torch.long).to(config.device)
    c_n_real = get_neutral_code(x_n.shape[0], x_n.dtype, config)
    c_e = generate_rand_code(x_n.size(0), config)
    # c_e_std_r, _ = torch.std_mean(c_e, dim=0)
    # c_e_std_r = torch.mean(c_e_std_r.detach()).item()

    x_fake_e = model.generator(x_n, c_e)

    adv_out_e = model.discriminator(x_fake_e, y_e)
    loss_adv_e = adv_loss(adv_out_e, 1)

    x_n_back = model.generator(x_fake_e, c_n_real)
    loss_cyc_n = torch.mean(torch.abs(x_n_back - x_n))

    # diversity sensitive loss
    loss_ds_e = None
    if config.lambda_ds_e > 0:
        c_e_2 = generate_rand_code(x_n.size(0), config)
        # else:
        #     c_e_2 = encode(model.encoder, sample.x_e_2, config)
        x_fake_e_2 = model.generator(x_n, c_e_2)
        x_fake_e_2 = x_fake_e_2.detach()
        c_e_diff = torch.abs(c_e_2.detach() - c_e.detach())
        c_e_diff = c_e_diff.view(c_e_diff.size(0), -1).mean(dim=1)
        c_e_diff = c_e_diff.mul(CODE_DIFF_EXP_FRACTION)  # scale to 1

        loss_ds_e_each = (torch.abs(x_fake_e - x_fake_e_2))
        loss_ds_e_each = loss_ds_e_each.view(loss_ds_e_each.size(0), -1).mean(dim=1)
        loss_ds_e = - torch.mean(c_e_diff * loss_ds_e_each)

    # encoder must be able to extract same rand_code from x_fake_e
    # generator must contain full expression information
    c_e_fake = encode(model.encoder, x_fake_e, config)
    c_e_r_std, c_e_r_m = torch.std_mean(c_e_fake, dim=0)
    c_e_r_std = torch.mean(c_e_r_std.detach()).item()
    c_e_r_m = torch.mean(c_e_r_m.detach()).item()
    loss_c_e = torch.mean(torch.abs(c_e_fake - c_e))
    # loss_c_e = F.mse_loss(c_e_fake - c_e) weaker than L1

    loss_adv_e = config.lambda_adv_e * loss_adv_e
    loss_cyc_n = config.lambda_cyc_n * loss_cyc_n
    # loss_adv_n = config.lambda_adv_n * loss_adv_n
    # loss_cyc_e = config.lambda_cyc_e * loss_cyc_e
    loss = loss_adv_e + loss_cyc_n
    if loss_ds_e is not None:
        loss_ds_e = config.lambda_ds_e * loss_ds_e
        loss = loss + loss_ds_e
        loss_ds_e = loss_ds_e.item()
    if loss_c_e is not None:
        loss_c_e = config.lambda_c_e * loss_c_e
        loss = loss + loss_c_e
        loss_c_e = loss_c_e.item()
    return loss, Munch(
        # total=loss.item(),
        # adv_n=loss_adv_n.item(),
        adv_e=loss_adv_e.item(),
        cyc_n=loss_cyc_n.item(),
        # cyc_e=loss_cyc_e.item(),
        c_e=loss_c_e,
        ds_e=loss_ds_e,
        c_e_r_m=c_e_r_m,
        c_e_r_std=c_e_r_std,
        # c_e_std_r=c_e_std_r
    )


def loss_GE_ENE(model, config, sample):
    """
    G and E loss of cycle 2: E to N to E
    """
    x_e = sample.x_e.to(config.device)

    y_n = torch.zeros(x_e.shape[0], dtype=torch.long).to(config.device)
    c_n_real = get_neutral_code(x_e.shape[0], x_e.dtype, config)

    x_fake_n = model.generator(x_e, c_n_real)

    adv_out_n = model.discriminator(x_fake_n, y_n)
    loss_adv_n = adv_loss(adv_out_n, 1)

    c_e = encode(model.encoder, x_e, config)
    c_e_x_std, c_e_r_m = torch.std_mean(c_e, dim=0)
    c_e_x_std = torch.mean(c_e_x_std.detach()).item()

    x_e_back = model.generator(x_fake_n, c_e)
    loss_cyc_e = torch.mean(torch.abs(x_e_back - x_e))
    loss_adv_n = config.lambda_adv_n * loss_adv_n
    loss_cyc_e = config.lambda_cyc_e * loss_cyc_e
    loss = loss_adv_n + loss_cyc_e
    return loss, Munch(
        adv_n=loss_adv_n.item(),
        cyc_e=loss_cyc_e.item(),
        c_e_x_std=c_e_x_std,
    )


def train_step(model, optims, config, sample, loss_sts, loss_funs):
    """
    Do backward and step.

    :param model: model
    :param optims: optims
    :param config: config
    :param loss_sts: loss status
    :param loss_funs: loss functions
    """
    loss_total = None
    for loss_fun in loss_funs:
        loss, losses = loss_fun(model, config, sample)
        if loss_total is None:
            loss_total = loss
        else:
            loss_total += loss
        losses_average(loss_sts, losses)
    for op in optims:
        op.zero_grad()
    loss_total.backward()
    for op in optims:
        op.step()


def train(config):
    print(config)
    timer = Timer(config.resume_iter, config.total_iter)
    cuda_count = torch.cuda.device_count()
    print(f"Number of GPUs: {cuda_count}")
    print(f"Using: {config.device}")
    print()

    sample_getter, sample_getter_test = create_sample_getter(config)

    model, model_s = create_model(config)

    optims = Munch()
    optims.generator = torch.optim.Adam(
        params=model.generator.parameters(),
        lr=config.lr,
        betas=[config.beta1, config.beta2],
        weight_decay=config.weight_decay)

    optims.discriminator = torch.optim.Adam(
        params=model.discriminator.parameters(),
        lr=config.lr,
        betas=[config.beta1, config.beta2],
        weight_decay=config.weight_decay)

    optims.encoder = torch.optim.Adam(
        params=model.encoder.parameters(),
        lr=config.lr_e,
        betas=[config.beta1, config.beta2],
        weight_decay=config.weight_decay)

    if config.test:
        config.total_iter = 1
    print("Start training...")

    if config.resume_iter > 0:
        load_model_all(config, model, model_s, optims, config.resume_iter)

    lambda_c_e_config = parse_lambda_config(config.lambda_c_e_config)
    config.lambda_c_e = calculate_lambda(config.resume_iter, *lambda_c_e_config)
    print(f"lambda_c_e_end_v:{lambda_c_e_config[3]}")
    print(f"lambda_c_e:{config.lambda_c_e}")

    lambda_ds_e_config = parse_lambda_config(config.lambda_ds_e_config)
    config.lambda_ds_e = calculate_lambda(config.resume_iter, *lambda_ds_e_config)
    print(f"lambda_ds_e:{config.lambda_ds_e}")

    lambda_cyc_e_config = parse_lambda_config(config.lambda_cyc_e_config)
    config.lambda_cyc_e = calculate_lambda(config.resume_iter, *lambda_cyc_e_config)
    print(f"lambda_cyc_e:{config.lambda_cyc_e}")

    # for computing the average of the loss for logging
    d_losses_avg = Munch(real_n=None,
                         real_e=None,
                         fake_n=None,
                         fake_e=None,
                         reg_n=None,
                         reg_e=None,
                         )
    g_losses_avg = Munch(total=None,
                         adv_n=None,
                         adv_e=None,
                         cyc_n=None,
                         cyc_e=None,
                         c_e=None,
                         ds_e=None,
                         c_e_r_m=None,
                         c_e_r_std=None,
                         c_e_x_std=None,
                         )
    for step in range(config.resume_iter + 1, config.total_iter + 1):
        config.step = step
        config.lambda_c_e = calculate_lambda(step, *lambda_c_e_config)
        config.lambda_ds_e = calculate_lambda(step, *lambda_ds_e_config)
        config.lambda_cyc_e = calculate_lambda(step, *lambda_cyc_e_config)

        # training ratio of cycle 1 : cycle 2 is 3 : 2
        sample1 = sample_getter.next_sample()
        train_step(model, [optims.discriminator], config, sample1, d_losses_avg, [loss_D_ENE, loss_D_NEN])
        train_step(model, [optims.encoder, optims.generator]
                   , config, sample1, g_losses_avg, [loss_GE_ENE, loss_GE_NEN])

        sample2 = sample_getter.next_sample()
        train_step(model, [optims.discriminator], config, sample2, d_losses_avg, [loss_D_ENE, loss_D_NEN])
        train_step(model, [optims.encoder, optims.generator]
                   , config, sample2, g_losses_avg, [loss_GE_ENE, loss_GE_NEN])

        sample3 = sample_getter.next_sample()
        train_step(model, [optims.discriminator], config, sample3, d_losses_avg, [loss_D_NEN])
        train_step(model, [optims.encoder, optims.generator]
                   , config, sample3, g_losses_avg, [loss_GE_NEN])

        # model_s will take the average of the last n back propagation
        # model_s is a stable version of model
        copy_model_average(model, model_s)

        timer.increase()
        if config.test or (step == 1) or (step % config.log_every == 0):
            log = generate_log(step, timer, d_losses_avg, g_losses_avg, config)
            print(log)

        # generate sample images
        if (step == 1) or (step % config.output_every == 0):
            # sample = sample_getter.next_sample()
            # sample2 = sample_getter.next_sample()
            generate_output(model_s, config, sample1, sample3, step, "train")
        if (step == 1) or step % (config.output_every * 2) == 0:
            generate_output_test(model_s, config, step, sample_getter_test)

        # save models
        if (step == 1) or step % config.save_every == 0:
            save_model_all(config, model, model_s, optims, step)
        if config.test:
            save_model_all(config, model, model_s, optims, step)
            load_model_all(config, model, model_s, optims, step)

    print("Training done!")
