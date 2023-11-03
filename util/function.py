import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import json
import os
from lib.loss import get_loss

def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight)
        try:
            m.bias.data.zero_()
        except:
            pass


def cat_score(score, score_pre, dim):
    if score_pre == '':
        i = 0
    else:
        i = 1
    if i == 0:
        score_pre = score
    else:
        score_pre = torch.cat([score_pre, score], dim=dim)
    return score_pre


def mean_frame(position, atten_tsm_rgb):
    score_rgb = ''
    for f in range(position.shape[0]):
        if f == 0:
            frame = atten_tsm_rgb[:, :position[f]]
        elif f == position.shape[0] - 1:
            frame = atten_tsm_rgb[:, position[f - 1]:]
        else:
            frame = atten_tsm_rgb[:, position[f - 1]: position[f]]
        score = torch.mean(frame, dim=1)
        score = score.unsqueeze(1)
        score_rgb = cat_score(score, score_rgb, 1)
    return score_rgb


def train_sample(args, samples, sample_lens, sample_labels, sample_mask, attgen, optimizer_filter, fcencoder, optimizer):
    fcencoder.train()
    attgen.train()
    flag_sample = True
    while flag_sample:
        optimizer.zero_grad()
        optimizer_filter.zero_grad()
        samples_s = fcencoder(samples, sample_lens)
        mean_feats = get_mean_feats(samples_s, sample_lens, args.pick_class_num * args.sample_num_per_class)

        # Given several query videos, infer attention mask using reference/sample videos.
        atten_frame_level = get_attention_mask(samples_s, sample_mask, args, args.pick_class_num, args.num_in, attgen)

        # print('Eu distance between temporal pooling batches and mean feature')
        loss_cls = get_loss(samples_s,
                            atten_frame_level.transpose(1, 2),
                            torch.eye(args.pick_class_num)[sample_labels],
                            sample_lens.repeat(args.sample_num_per_class),
                            mean_feats,
                            args.pick_class_num,
                            distance=args.distance)

        loss = loss_cls
        loss.backward()
        optimizer.step()
        optimizer_filter.step()
        print('\r', 'sample_loss:', round(float(loss), 3), end='', flush=True)
        if float(loss) < 0.01:
            flag_sample = False


def test_sample(sample_network, samples):
    sample_network.eval()
    samples_s = sample_network(samples)
    return samples_s


def get_attention_mask(samples, sample_mask, args, num_class, num_in, comparator=[]):
    mask = get_tsm_cos(sample_mask, args, num_class, norm=False)
    if num_in == 1:
        attention_mask = get_attention_mask_1(mask, samples, args, num_class)
    elif num_in == 2:
        attention_mask = get_attention_mask_2(mask, samples, args, num_class, comparator)
    elif num_in == 4:
        attention_mask = get_attention_mask_4(mask, samples, args, num_class, comparator)
    elif num_in == 6:
        attention_mask = get_attention_mask_6(mask, samples, args, num_class, comparator)

    return attention_mask


def get_attention_mask_4(mask, samples, args, num_class, comparator=[]):

    samples_flow = samples[:, :, :args.num_dim]
    samples_rgb = samples[:, :, args.num_dim:]

    # mask = get_tsm_cos(sample_mask, batch_mask, args, num_class, mode, norm=False)

    tsm_rgb = get_tsm_cos(samples_rgb, args, num_class, norm=True)
    tsm_masked_rgb = tsm_rgb * mask

    tsm_flow = get_tsm_cos(samples_flow, args, num_class, norm=True)
    tsm_masked_flow = tsm_flow * mask

    # [bs, num_class, L]
    atten_tsm_rgb, _ = torch.max(tsm_masked_rgb, dim=-1, keepdim=True)
    atten_tsm_flow, _ = torch.max(tsm_masked_flow, dim=-1, keepdim=True)

    ssm_rgb = get_tsm_cos(samples_rgb, args, num_class, norm=False)
    ssm_masked_rgb = ssm_rgb * mask
    atten_ssm_rgb, _ = torch.max(ssm_masked_rgb, dim=-1, keepdim=True)

    ssm_flow = get_tsm_cos(samples_flow, args, num_class, norm=False)
    ssm_masked_flow= ssm_flow * mask
    atten_ssm_flow, _ = torch.max(ssm_masked_flow, dim=-1, keepdim=True)

    # print(atten_tsm.size(), atten_ssm.size())
    atten_frame_level = comparator(torch.cat([atten_ssm_rgb, atten_ssm_flow, atten_tsm_rgb, atten_tsm_flow], dim=-1))

    return atten_frame_level


def get_tsm_cos(samples_enc, args, pick_class_num, norm=False):
    # each batch sample link to every samples to calculate similarities
    samples_a = samples_enc.unsqueeze(0).repeat(args.sample_num_per_class * pick_class_num, 1, 1, 1)
    samples_b = samples_enc.unsqueeze(0).repeat(args.sample_num_per_class * pick_class_num, 1, 1, 1)

    samples_b = torch.transpose(samples_b, 0, 1)

    if norm:
        samples_a = samples_a / (samples_a.norm(dim=-1)[..., None].repeat(1, 1, 1, samples_enc.size(-1)) + 1e-5)
        samples_b = samples_b / (samples_b.norm(dim=-1)[..., None].repeat(1, 1, 1, samples_enc.size(-1)) + 1e-5)

    tsm = torch.matmul(samples_a, samples_b.transpose(-2, -1))

    return tsm


def save_model_alone(path, epoch, model, optimizer, name):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}

    save_file = os.path.join(path, 'model_' + name + '.pth')
    save_checkpoint(state, save_file)
    print('save model: %s' % save_file)
    return save_file


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_mean_feats(samples, sample_lens, num, device=torch.device("cuda")):
    mean_sample = torch.zeros(0).to(device)
    for i in range(len(samples)):
        sam_feat = samples[i, :sample_lens[i]]
        mean_sample = torch.cat([mean_sample, torch.mean(sam_feat, dim=0, keepdim=True)], dim=0)
    return mean_sample


def bi_inter(data, window):
    if data.ndim == 3:
        data = data.unsqueeze(0)
        cha, cls = data.shape[2:]
        data = F.interpolate(data, size=[window, cls], mode='bilinear', align_corners=True)
        data = data.squeeze(0)
    elif data.ndim == 4:
        cha, cls = data.shape[2:]
        data = F.interpolate(data, size=[window, cls], mode='bilinear', align_corners=True)
    return data