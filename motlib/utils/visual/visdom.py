from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging
from PIL import Image
import json
from json import JSONDecodeError
from .visual_base import VisualBase
import util.misc as comm
from motlib.utils import PathManager


class VisualVisdom(VisualBase):
    def init(self):
        self._file_name = self._env_name + '.json'
        self._file_dir = self._output_dir / self._file_name

        self.online = False
        logger = logging.getLogger(__name__)

        try:
            from visdom import Visdom
            self._writer = Visdom(port=self._port, env=self._env_name, server=self._host,
                                  log_to_filename=self._file_dir,
                                  raise_exceptions=True)
        except ImportError as e:
            logger.error(e)
            logger.error("This module requires visdom package. "
                         "Please install it with command: pip install visdom")
            self._writer = None
        except (ConnectionError, ConnectionRefusedError) as e:
            logger.error(e)
            self._writer = Visdom(port=self._port, env=self._env_name, server=self._host,
                                  log_to_filename=self._file_dir, offline=True,
                                  raise_exceptions=True)
            logger.warning("Visdom is installed, but no server connection. "
                          "All requests are logged to file {}".format(self._file_dir))
        else:
            self.online = True
            logger.info('Visdom is available. Using Visdom to display visual data on the web page.')
        finally:
            self._win_id = {}
            self._win_id.update(self._load_win_id())

    def _win_registor(self, win_name, line_name, X, Y):
        self._win_id[win_name] = self._writer.line(
            X=X,
            Y=Y,
            opts=dict(
                legend=[line_name],
                markers=True,
                title=win_name,
                markersize=3,
                showlegend=True
            )
        )

    def add_scalar(self, win_name, line_name, X, Y):
        line_name = self._legend_wrap(line_name)

        if not isinstance(X, (np.ndarray, list, tuple)):
            X = np.asarray([X])
        if not isinstance(Y, (np.ndarray, list, tuple)):
            Y = np.asarray([Y])

        if win_name in self._win_id:
            if self.online:
                if not self._writer.win_exists(self._win_id[win_name]):
                    self._win_registor(win_name, line_name, X, Y)
                    return
            self._writer.line(
                X=X,
                Y=Y,
                win=self._win_id[win_name],
                update='append',
                name=line_name,
                opts=dict(showlegend=True)
            )
        else:
            self._win_registor(win_name, line_name, X, Y)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):

        for line_name in tag_scalar_dict.keys():
            self.add_scalar(main_tag, line_name, global_step, tag_scalar_dict[line_name])

    def flush(self):
        if not comm.is_main_process():
            return

        if self.online:
            self._writer.save([self._env_name])

            if not self._copy:
                return
            try:
                if self._file_dir.exists():
                    PathManager.rm(self._file_dir)
                PathManager.mv_file(self._env_dir / self._file_name, self._output_dir)
            except FileExistsError:
                pass

    def close(self):
        self.flush()

    def _load_win_id(self):
        if not self._file_dir.exists():
            return{}
        win_id = {}
        try:
            data = PathManager.load(self._file_dir, False, False)['jsons']
            for k, v in data.items():
                if 'id' in v and 'title' in v:
                    wid = v['id']
                    win_name = v['title']
                    if win_name in win_id:
                        logger = logging.getLogger(__name__)
                        logger.info('%s' % data)
                        logger.info('%s' % win_id)
                        logger.info('%s' % v)
                        raise RuntimeError
                    else:
                        win_id[win_name] = wid
        except JSONDecodeError:
            with PathManager.open(self._file_dir, 'r') as f:
                f = f.readlines()
            for info in f:
                info = json.loads(info)
                for i in info:
                    if isinstance(i, dict):
                        if 'win' in i and 'layout' in i and 'title' in i['layout']:
                            win_name = i['layout']['title']
                            if win_name in win_id:
                                assert win_id[win_name] == i['win'], 'Old name {} id {} New id {}'.format(win_name,
                                                                                                          win_id[
                                                                                                              win_name],
                                                                                                          i['win'])
                            else:
                                win_id[win_name] = i['win']
        return win_id

    @staticmethod
    def _read_img(img_dir, h=None, w=None):
        img = Image.open(img_dir).convert('RGB')
        if h is not None and w is not None:
            img = img.resize([w, h], Image.BILINEAR)
        return np.asarray(img)

    def image(self, img, *, caption='', store_history=True, title=None, win=None, h=None, w=None):
        if win is None and title is not None:
            win = title
        if isinstance(img, str):
            img = self._read_img(img, h=h, w=w)
        win = self._writer.image(img, win=win, opts=dict(caption=caption, store_history=store_history, title=title))
        return win

    def images(self, imgs, *, caption='', store_history=True, title=None, win=None, h=None, w=None):
        if win is None and title is not None:
            win = title
        if isinstance(imgs, (list, tuple)):
            if isinstance(imgs[0], str):
                imgs = [self._read_img(img_dir=img_dir, h=h, w=w) for img_dir in imgs]
        win = self._writer.image(imgs, win=win, opts=dict(caption=caption, store_history=store_history, title=title))
        return win
