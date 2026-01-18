# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/8, 2020/9/16, 2021/7/1, 2021/7/11
# @Author : Zhen Tian, Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyuwuxinn@gmail.com, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

from gc import DEBUG_SAVEALL
from unittest import TestCase
import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()

    def _change_feat_format(self, ds_list):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        train, val, test = super()._change_feat_format(ds_list)
        print("instance of ds", type(train))

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        final_ds_list = []
        for ds in [train, val, test]:
            final_ds_list.append(self.augment_one_dataset(ds))
        return final_ds_list

    def _aug_presets(self, ds):
        """
        Set sequential-augmentation metadata on `ds`:
        - create ds.<field>_list_field names
        - register their field properties (type/source/seqlen) in ds.field2type/field2seqlen/field2source
        - register ds.item_list_length_field
        """
        list_suffix = ds.config["LIST_SUFFIX"]
        max_len = ds.config["MAX_ITEM_LIST_LENGTH"]  # more robust than self.max_item_list_len [web:144]

        for field in ds.inter_feat:
            if field == ds.uid_field:
                continue

            list_field = field + list_suffix
            setattr(ds, f"{field}_list_field", list_field)

            ftype = ds.field2type[field]

            # token -> token_seq, float -> float_seq
            if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                list_ftype = FeatureType.TOKEN_SEQ
            else:
                list_ftype = FeatureType.FLOAT_SEQ

            # If original field already sequence-like, store (max_len, original_seqlen),
            # else store max_len. This mirrors RecBole SequentialDataset behavior. [web:124]
            if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                list_len = (max_len, ds.field2seqlen[field])
            else:
                list_len = max_len

            ds.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        # record sequence length field (e.g., "item_length") [web:144]
        ds.set_field_property(
            ds.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    
    def augment_one_dataset(self, ds):
        """Return a new Dataset with sequential augmentation applied."""
        # Use ds’ metadata but reuse the same augmentation presets in `self`
        # (or move _aug_presets/_check_field onto ds if you prefer).
        self._aug_presets(ds)
        ds._check_field("uid_field", "time_field")

        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        ds.sort(by=[ds.uid_field, ds.time_field], ascending=True)

        last_uid = None
        item_list_index, target_index, item_list_length = [], [], []
        seq_start = 0

        for i, uid in enumerate(ds.inter_feat[ds.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_data = ds.inter_feat[target_index]
        new_dict = {ds.item_list_length_field: torch.tensor(item_list_length)}

        new_length = len(target_index)
        for field in ds.inter_feat:
            print()
            print("ds.inter_feat!!!", ds.inter_feat)
            print()

            if field == ds.uid_field:
                continue
            list_field = getattr(ds, f"{field}_list_field")
            list_len = ds.field2seqlen[list_field]

            shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
            if (ds.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                and field in ds.config["numerical_features"]):
                shape += (2,)

            new_dict[list_field] = torch.zeros(shape, dtype=ds.inter_feat[field].dtype)

            value = ds.inter_feat[field]
            for j, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                new_dict[list_field][j][:length] = value[index]

        new_data.update(Interaction(new_dict))
        return ds.copy(new_data)


    # def data_augmentation(self):
    #     """Augmentation processing for sequential dataset.

    #     E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
    #     then after augmentation, we will generate three cases.

    #     ``u1, <i1> | i2``

    #     (Which means given user_id ``u1`` and item_seq ``<i1>``,
    #     we need to predict the next item ``i2``.)

    #     The other cases are below:

    #     ``u1, <i1, i2> | i3``

    #     ``u1, <i1, i2, i3> | i4``
    #     """
    #     self.logger.debug("data_augmentation")

    #     self._aug_presets()

    #     self._check_field("uid_field", "time_field")
    #     max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
    #     self.sort(by=[self.uid_field, self.time_field], ascending=True)
    #     last_uid = None
    #     uid_list, item_list_index, target_index, item_list_length = [], [], [], []
    #     seq_start = 0
    #     for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
    #         if last_uid != uid:
    #             last_uid = uid
    #             seq_start = i
    #         else:
    #             if i - seq_start > max_item_list_len:
    #                 seq_start += 1
    #             uid_list.append(uid)
    #             item_list_index.append(slice(seq_start, i))
    #             target_index.append(i)
    #             item_list_length.append(i - seq_start)

    #     uid_list = np.array(uid_list)
    #     item_list_index = np.array(item_list_index)
    #     target_index = np.array(target_index)
    #     item_list_length = np.array(item_list_length, dtype=np.int64)

    #     new_length = len(item_list_index)
    #     new_data = self.inter_feat[target_index]
    #     new_dict = {
    #         self.item_list_length_field: torch.tensor(item_list_length),
    #     }

    #     for field in self.inter_feat:
    #         if field != self.uid_field:
    #             list_field = getattr(self, f"{field}_list_field")
    #             list_len = self.field2seqlen[list_field]
    #             shape = (
    #                 (new_length, list_len)
    #                 if isinstance(list_len, int)
    #                 else (new_length,) + list_len
    #             )
    #             if (
    #                 self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
    #                 and field in self.config["numerical_features"]
    #             ):
    #                 shape += (2,)
    #             new_dict[list_field] = torch.zeros(
    #                 shape, dtype=self.inter_feat[field].dtype
    #             )

    #             value = self.inter_feat[field]
    #             for i, (index, length) in enumerate(
    #                 zip(item_list_index, item_list_length)
    #             ):
    #                 new_dict[list_field][i][:length] = value[index]

    #     new_data.update(Interaction(new_dict))
    #     self.inter_feat = new_data

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.inter_feat[self.item_list_length_field] = self.inter_feat[
            self.item_id_list_field
        ].agg(len)

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()
