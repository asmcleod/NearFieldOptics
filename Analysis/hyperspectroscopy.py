import numpy as np
import os
import copy
import time
from matplotlib import pyplot as plt
from common import misc
from common import numerics as num
from common.baseclasses import AWA
import cv2

figsize=(12, 6)

cmplx_dtypes = [complex,np.complex64]

cv2_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-6)

class ImageSynchronizer:

    warp_modes = dict(homography=cv2.MOTION_HOMOGRAPHY,
                      affine=cv2.MOTION_AFFINE)
    warp_methods = list(warp_modes.keys())
    all_methods = ['correlation'] + warp_methods

    def find_warp(self,ref_image,distorted_image,method='homography'):

        assert method in self.warp_methods
        if method in ['homography']:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            inv_warp_matrix = np.eye(3, 3, dtype=np.float32)
        elif method in ['euclidean','affine']:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            inv_warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif method in ['translation']:
            warp_matrix = np.eye(2, 2, dtype=np.float32)
            inv_warp_matrix = np.eye(2, 2, dtype=np.float32)
        warp_mode = self.warp_modes[method]

        # Settings for the number of iterations and error of the image transformation
        # Compute time becomes significant above these values
        distorted_image = np.array(distorted_image).astype(np.float32)
        ref_image = np.array(ref_image).astype(np.float32)
        self.cc1, warp_matrix = cv2.findTransformECC(distorted_image, ref_image,
                                                     warp_matrix, warp_mode, criteria = cv2_criteria)
        self.cc2, inv_warp_matrix = cv2.findTransformECC(ref_image, distorted_image,
                                                         inv_warp_matrix, warp_mode, criteria=cv2_criteria)
        # cc is a statistical measure of the quality of the overlap

        return warp_matrix, inv_warp_matrix

    def perform_perspective(self,img):
        import cv2
        size = img.shape
        try: warp_matrix = self._warp_matrix
        except AttributeError:
            raise RuntimeError('Run the `find_warp` method first to produce and store a warp matrix!')
        # optional 1,0 mask to remove non overlapping regions
        img_input = np.array(img).astype(np.float32)
        img_warped = cv2.warpPerspective(img_input, warp_matrix,
                                           (size[1], size[0]),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue = np.nan)
        # Affine Transform is called cv2.warpAffine(matrix,warp,(size[1,size[0]]))

        if isinstance(img,AWA):
            img_warped = AWA(img_warped,adopt_axes_from=img)

        return img_warped

    def perform_affine(self,img):
        import cv2
        size = img.shape
        try: warp_matrix = self._warp_matrix
        except AttributeError:
            raise RuntimeError('Run the `find_warp` method first to produce and store a warp matrix!')
        # optional 1,0 mask to remove non overlapping regions
        img_input = np.array(img).astype(np.float32)
        img_warped = cv2.warpAffine(img_input, warp_matrix,
                                           (size[1], size[0]),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue = np.nan)
        # Affine Transform is called cv2.warpAffine(matrix,warp,(size[1,size[0]]))

        if isinstance(img,AWA):
            img_warped = AWA(img_warped,adopt_axes_from=img)

        return img_warped

    @staticmethod
    def find_cross_correlation_shift(ref_image, distorted_image):

        from common import numerical_recipes as numrec
        ref_image = np.array(ref_image) # Just to make sure we use axes that are pixel values
        distorted_image = np.array(distorted_image)
        CC = numrec.CrossCorrelate(ref_image, distorted_image).real
        dx,dy = CC.locate(CC.max())[0]

        return dx,dy

    def perform_cross_correlation_shift(self,img):

        from scipy import ndimage

        img = AWA(img)
        try: dx,dy = self._dxy_shift
        except AttributeError:
            raise RuntimeError('Run the `find_cross_correlation_shift` method first to produce and store a dx,dy shift!')
        img_warped = ndimage.shift(img, (dx, dy), cval = np.nan) # `nan` will denote where we had to shift beyond available data
        img_warped = AWA(img_warped, adopt_axes_from=img)

        return img_warped

    def __init__(self,ref_image,distorted_image,method):

        self.warp_performers = dict(homography=self.perform_perspective,
                                    affine=self.perform_affine)

        if method in self.warp_methods:
            self._warp_matrix,self._inv_warp_matrix = self.find_warp(ref_image,distorted_image,
                                                                   method=method)
            self.transformer = self.warp_performers[method]
        elif method == 'correlation':
            self._dxy_shift = self.find_cross_correlation_shift(ref_image, distorted_image)
            self.transformer = self.perform_cross_correlation_shift
        else:
            raise ValueError('Method "%s" must be one of %s!'%(method,self.all_methods))
        self.method = method

    def get_valid_mask(self,distorted_image):

        original_valid_mask = np.ones(distorted_image.shape)
        valid_mask = self.transformer(original_valid_mask)
        valid_mask[np.isnan(valid_mask)] = 0

        return valid_mask.astype(bool)

    def __call__(self,img,**kwargs): return self.transformer(img,**kwargs)


class Hyperspectroscopy(object):

    preferred_formats = ['.txt','.exr','.tiff'] # in order of preference
    clim_nstds = 2
    debug = True

    def __init__(self, dirs=['AutoRepScan overview - 30x18 microns - S3 @ 12.5-7.7 microns x 11 (6-16)'],
                 channels=None, directions=['Forward', 'Backward'],
                 keys=np.linspace(12.5, 7.7, 11)):

        if channels is None:
            channels = ['Topo', 'S3', 'P3']
        import copy

        # Check keys and dirs
        assert hasattr(keys,'__len__')
        if len(dirs) > 1:
            assert hasattr(keys[0],'__len__') and len(keys) == len(dirs), \
                'Provide a key list in `keys` for each of the directories you supplied in `dirs` (you provided %i!=%i).'\
                %(len(keys), len(dirs))
        else:
            keys =[keys]

        self.dirs = dirs
        self.channels = channels
        self.keys = np.array(list(misc.flatten(keys)))  # Join all the individual key lists
        self.directions = directions

        # Make holder for image data in all directions
        images_in_direction = dict(zip(channels, [{} for c in channels]))  # all channels
        self.images = dict(zip(directions,
                                 [copy.deepcopy(images_in_direction) for direction in directions]))  # all directions
        self.file_paths = copy.deepcopy(self.images)

        # Load desired channel data
        shape = None # we'll be watching to see every image has the same shape
        for keylist, dirname in zip(keys, dirs):

            # Start loading each of the channels
            for channel in channels:

                # Start loading each of the directions
                for direction in directions:
                    channeldir = os.path.join(dirname, channel, direction)
                    assert os.path.isdir(channeldir), 'directory "%s" must exist!' % channeldir

                    all_filenames = os.listdir(channeldir)
                    # Keep only one image file for each "prefix" even if multiple extensions are found
                    extensions_dict = {}
                    for filename in all_filenames:
                        prefix,ext = os.path.splitext(filename)
                        if ext not in self.preferred_formats: continue
                        if prefix in extensions_dict: # we already found a file with this prefix
                            previous_ext = extensions_dict[prefix]
                            preferred = self.preferred_formats.index(ext) < self.preferred_formats.index(previous_ext)
                            if not preferred:
                                print('Skipping file "%s" because preferred extension "%s" was found.' % (filename, previous_ext))
                                continue
                        extensions_dict[prefix] = ext

                    # Rebuild and sort (ascending) the image file paths now that they've been filtered
                    image_files = sorted([prefix + extensions_dict[prefix] for prefix in extensions_dict])
                    assert len(image_files)==len(keylist),\
                                'Number of images in "%s" should match (length of keys) = %i, instead found files: \n%s'\
                                % (channeldir, len(keylist) , '\n'.join(image_files))

                    # Load an image for each key
                    for key, image_file in zip(keylist, image_files):
                        file_path = os.path.join(channeldir,image_file)
                        img = self.load_image( file_path )
                        if shape is None: shape = img.shape
                        else:
                            if img.shape != shape: img=img.T
                            assert img.shape == shape,'Shape of image file "%s" must match %s!'%(image_file, str(shape))
                        self.images[direction][channel][key] = img
                        self.file_paths[direction][channel][key] = file_path

        self.valid_mask = (1 + np.zeros(shape)).astype(bool)

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

    def __getattribute__(self,attr):

        if attr in ('xy','XY','AWA'):
            direction = self.directions[0]
            channel=self.channels[0]
            key = self.keys[0]
            img = self.images[direction][channel][key]

            if attr=='xy': return img.axes
            elif attr=='XY': return img.axis_grids
            elif attr=='AWA': return img

        else: return super().__getattribute__(attr)

    def get_valid_xy_range(self):

        valid_xspace = np.any(self.valid_mask,axis=1)
        valid_yspace = np.any(self.valid_mask,axis=0)

        return valid_xspace,valid_yspace

    def get_valid_mask(self):

        return AWA(self.valid_mask, adopt_axes_from=self.AWA)

    def plot_images(self,channel='S3',direction='Forward',grid=True):

        for key in self.keys:
            img=self.images[direction][channel][key]

            plt.figure()
            img.plot()
            plt.gca().set_aspect('equal')
            plt.title('%s, %s, key=%s'%(direction,channel,key))
            if grid: plt.grid(ls='--',color='w')

    def commit_changes(self):
        if self.debug: print('Persisting the processed data from `self.images_processed` to `self.images`!')
        self.images = copy.deepcopy(self.images_processed)
        self.valid_mask = copy.deepcopy(self.valid_mask_processed)

    def update_valid_mask(self,valid_mask = None,
                          use_channel='S3'):
        """Further udate the internal mask (or take from provided `valid_mask` to exclude image regions that are unknown (nan)."""

        if valid_mask is None: valid_mask = self.valid_mask

        for key in self.keys:
            for direction in self.directions:
                img = self.images[direction][use_channel][key]
                assert valid_mask.shape == img.shape
                valid_mask *= np.isfinite(img)

    @classmethod
    def load_image(cls,image_file):

        import os
        prefix,ext =  os.path.splitext(image_file)

        if ext == '.tiff':
            import pspylib as ps
            a = ps.TiffReader(image_file)  # loading object
            h = a.data.scanHeader.scanHeader  # header information
            z = a.data.scanData.ZData.reshape((h['height'][0], h['width'][0]))  # imaging data (2D)
            x = np.linspace(0, h['scanSizeWidth'][0], h['width'][0])  # x-axis data
            y = np.linspace(0, h['scanSizeHeight'][0], h['height'][0])  # y-axis data

            img = AWA(z.T, axes=[x, y], axis_names=['X (microns)', 'Y (microns)'])

        elif ext=='.txt':

            img = np.loadtxt(image_file)
            img = img[::-1] # Gwyddion exports have a habit of top-down counting of rows
            img = AWA(img.T, axis_names=['X (microns)', 'Y (microns)'])

        elif ext=='.exr':

            import cv2
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Thing to make openexr work,
            img = cv2.imread('Topo_0.6_40K_Left.exr',cv2.IMREAD_UNCHANGED)
            img = AWA(img, axis_names=['X (microns)', 'Y (microns)'])

        else:
            raise NotImplementedError('Image extension "%s" not yet supported...'%ext)

        if cls.debug:  print('Loaded data from file "%s".' % image_file)

        return img

    @staticmethod
    def sanitized_image(img,sanitize_where=None):

        assert isinstance(img,np.ndarray)

        # Get rid of `nan` in favor of finite median
        img = img.copy()
        if sanitize_where is None: sanitize_where = np.zeros(img.shape).astype(bool)
        sanitize_where += np.isnan(img)
        img[sanitize_where] = np.median(img[~sanitize_where])

        return img

    @staticmethod
    def serpent_unwrap(img, mask=None,fast_axis=0):

        if fast_axis == 1:
            iter_axis = 0  # if fast axis is y, then we will go column by column iterating on x
        else:
            iter_axis = 1  # if fast axis is x, then we will go row by row iterating on y

        img_vals = []
        Is, Js = AWA(np.array(img)).axis_grids
        Ivals = []
        Jvals = []
        for i in range(img.shape[iter_axis]):
            slices = [slice(None) for i in range(img.ndim)]
            slices[iter_axis] = i
            slices = tuple(slices)

            img_line = np.array(img[tuple(slices)])
            mask_line = np.array(mask[tuple(slices)]).astype(bool)
            Iline = np.array(Is[tuple(slices)])
            Jline = np.array(Js[tuple(slices)])

            # The 'serpent' part means we flip the line whenever we are at an odd row
            img_line = img_line[::(-1) ** i]
            mask_line = mask_line[::(-1) ** i]
            Iline = Iline[::(-1) ** i]
            Jline = Jline[::(-1) ** i]

            # serpent-unwrap only with the values under mask
            img_vals = np.append(img_vals, img_line[mask_line])
            Ivals = np.append(Ivals, Iline[mask_line]).astype(int)
            Jvals = np.append(Jvals, Jline[mask_line]).astype(int)

        return img_vals, Ivals, Jvals

    @classmethod
    def unwrap_image(cls,img,fast_axis=0,look_back=10,
                     discont=np.pi,thresh=0.75,mask=None):

        if mask is None:  mask = np.full(img.shape,1)

        # unwrap based on comparison to previous entries
        img_vals, Ivals, Jvals = cls.serpent_unwrap(img, fast_axis=fast_axis,
                                                    mask=mask)
        for i, img_val in enumerate(img_vals):
            if i == 0: continue
            back_ind = np.max((0, i - look_back))
            preval = np.median(img_vals[back_ind:i])
            if img_val - preval > thresh * discont:
                img_vals[i] -= discont
            elif img_val - preval < -thresh * discont:
                img_vals[i] += discont

        new_img = img.copy()
        for i, j, val in zip(Ivals, Jvals, img_vals):
            new_img[i, j] = val
        new_img = AWA(new_img, adopt_axes_from=img)

        return new_img

    def unwrap_images(self, channels=['P3'],
                      directions = None, keys=None,
                      fast_axis=0,
                       look_back=10,discont=np.pi,thresh=0.75,
                       plot=False,plot_clim=None):

        assert discont>0 and thresh>0.5

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        if directions is None: directions = self.directions # default to all
        if keys is None: keys = self.keys #default to all

        for direction in directions:
            for channel in channels:
                target_dict = self.images_processed[direction][channel]

                for key in keys:
                    img = target_dict[key]
                    if self.debug:
                        print('Unwrapping image (direction=%s, channel=%s, key=%s) wherever jump exceeds %i%% of discont=%1.2f'\
                              % (direction, channel, key, 100*thresh, discont))

                    if plot:
                        plt.figure(figsize=figsize)
                        plt.subplot(121)
                        img.plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('Before unwrapping: %s' % str((direction, channel, key)))
                        m=np.median(img[np.isfinite(img)])
                        if plot_clim is not None: plt.clim(*plot_clim)
                        else: plt.clim(m-discont,m+discont)

                    new_img = self.unwrap_image(img,fast_axis=fast_axis,look_back=look_back,
                                                discont=discont,thresh=thresh,mask=self.valid_mask)

                    if plot:
                        plt.subplot(122)
                        new_img.plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('After flattening: %s' % str((direction, channel, key)))
                        m=np.median(img[np.isfinite(img)])
                        if plot_clim is not None: plt.clim(*plot_clim)
                        else: plt.clim(m-discont,m+discont)

                    target_dict[key] = new_img

    @staticmethod
    def flatten_image(img,fast_axis=0,
                      poly_order=1,poly_divide=False,
                      mask=None,fourier_highpass=None,
                      line_poly=None,
                      median_of_diffs_flatten=False,
                      median_flatten=False):

        from common import numerical_recipes as numrec
        img = copy.copy(img) # Don't modify the original
        if mask is None: mask = np.full(img.shape,1)
        mask *= np.isfinite(img)
        mask = np.array(mask).astype(bool)

        if line_poly: assert isinstance(line_poly, int) and line_poly >= 0

        # Overall leveling
        if poly_order:
            polysurf = numrec.PolynomialFitImage(img, order=poly_order,
                                                 mask=mask,
                                                 full_output=False)
            if poly_divide:
                img /= polysurf
            else:
                img -= polysurf

        if fourier_highpass:
            img = numrec.FourierFilterImage(img, fmin=fourier_highpass)  # Tune `fmin` for more or less flatness

        # The following procedures are sensitive to the fast axis
        if fast_axis == 1:
            img_view = img  # If fast axis is y, we want to iterate column by column of img which is 0th dimension
        else:
            img_view = img.T  # If fast axis is x, we want to iterate row by row of img which is 1st dimension
            mask = mask.T

        #---- ALl the line-by-line leveling
        for i,line,mline in zip(range(img_view.shape[0]),
                                img_view,mask):
            if not np.any(mline): continue

            if line_poly:
                x = np.arange(len(line))
                p = np.polyfit(x[mline], line[mline], deg=line_poly)
                poly = np.polyval(p, x)

                if poly_divide:
                    line /= poly
                else: line -= poly

            if median_of_diffs_flatten:
                if i == 0: pass
                else:
                    mline = mask[i]*mask[i-1] # mutual mask
                    diff = img_view[i] - img_view[i - 1]
                    if np.any(mline):
                        img_view[i] -= np.median( diff[mline] )  # This is the "median of differences" flattening scheme

            elif median_flatten: line -= np.median(line[mline])

        return img

    def flatten_images(self, channels=['Topo'],directions=None,keys=None,
                       poly_order=3, poly_mask=None, valid_data=True, poly_divide=False,
                       fourier_highpass=.01,
                       median_of_diffs_flatten=False,
                       median_flatten=False,
                       line_poly=None,
                       fast_axis=0,
                       plot=False,plot_mask=False):
        """
        This is mostly used to flatten topographies before using them for synchronization.
        But could also be used to flatten a near-field amplitude or phase
        :param channels:
        :param poly_order:
        :param poly_mask:
        :param fourier_highpass:
        :param median_of_diffs_flatten:
        :param plot:
        :return:
        """

        if self.debug: print('Flattening images!')

        assert fast_axis in (0,1)
        assert False in (median_of_diffs_flatten,median_flatten), 'Only one of these two should be true!'

        if directions is None: directions = self.directions

        mask = None
        if valid_data: mask = self.valid_mask
        if poly_mask is not None: mask = mask*poly_mask

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        for direction in directions:
            for channel in channels:
                target_dict = self.images_processed[direction][channel]

                if keys is None: keys = list(target_dict.keys())

                for key in keys:
                    img = target_dict[key]

                    if self.debug:
                        print('Flattening image: direction=%s, channel=%s, key=%s' % (direction, channel, key))

                    if plot:
                        plt.figure(figsize=figsize)
                        plt.subplot(121)
                        img.plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('Before flattening: %s' % str((direction, channel, key)))

                        keep=self.valid_mask_processed.astype(bool)*np.isfinite(img)
                        m,s = np.median(img[keep]),np.std(img[keep])
                        plot_clim = (m-self.clim_nstds*s,
                                     m+self.clim_nstds*s)
                        plt.clim(*plot_clim)

                    img_flattened = self.flatten_image(img,fast_axis=fast_axis,
                                                      poly_order=poly_order,poly_divide=poly_divide,
                                                      mask=mask,fourier_highpass=fourier_highpass,
                                                      line_poly=line_poly,
                                                      median_of_diffs_flatten=median_of_diffs_flatten,
                                                      median_flatten=median_flatten)

                    if plot:
                        plt.subplot(122)
                        img = img_flattened
                        img.plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('After flattening: %s' % str((direction, channel, key)))

                        keep=self.valid_mask_processed.astype(bool)*np.isfinite(img)
                        m,s = np.median(img[keep]),np.std(img[keep])
                        plot_clim = (m-self.clim_nstds*s,
                                     m+self.clim_nstds*s)
                        plt.clim(*plot_clim)

                        if plot_mask:
                            mask_img = AWA(mask.astype(float).copy(),
                                          adopt_axes_from = img)
                            mask_img[mask_img==0] = np.nan
                            mask_img *=  plt.gci().get_clim()[1]
                            mask_img.plot(colorbar=False,alpha=float(plot_mask))
                            plt.gca().set_aspect('equal')
                            plt.clim(*plot_clim)

                    target_dict[key] = img_flattened

    def synchronize_images_to_ref_key(self,
                                      ref_key=None, ref_channel='Topo',
                                      direction='Forward',
                                      method = 'homography',
                                      use_valid_data=True,
                                      plot=False,plot_clim=None):
        """
        The idea here is to synchronize all images within a single direction using `use_channel` and targeting `ref_key`.
        This only applies within one direction.
        It is the job for a different function to synchronize directions, e.g. backward to forward.
        :param self:
        :param ref_channel:
        :param direction:
        :param ref_key:
        :param plot:
        :return:
        """

        import copy

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        # When synchronizing, perhaps we want to consider only data in the valid space
        if use_valid_data: sanitize_where = ~(self.valid_mask_processed.astype(bool))
        else: sanitize_where=None

        # Get the reference image
        if ref_key is None:
            ref_key = self.keys[0]
        if self.debug: print('Synchronizing %s images to channel %s, reference key %s!'%(direction,ref_channel,ref_key))
        ref_image = self.images_processed[direction][ref_channel][ref_key]
        ref_image = self.sanitized_image(ref_image,
                                             sanitize_where=sanitize_where) # sanitize in invalid data regions and places of `nan`

        # Loop on keys
        self.image_synchronizers = {}
        for key in self.keys:
            if key is ref_key: continue
            if self.debug: print('Synchronizing key=%s...' % key)

            # Get the image from which we will find the drift correction
            distorted_image = self.images_processed[direction][ref_channel][key].copy()
            distorted_image = self.sanitized_image(distorted_image,
                                                   sanitize_where=sanitize_where) # sanitize in invalid data regions and places of `nan`

            if plot:
                plt.figure(figsize=figsize)
                plt.subplot(121)
                img = distorted_image-ref_image
                (img).plot(colorbar=False)
                plt.gca().set_aspect('equal')
                plt.title('Diff before synchronizing: %s' % str((direction, ref_channel, key)))
                plt.grid(ls='--', color='w')
                if plot_clim is None:
                    keep=self.valid_mask_processed.astype(bool)*np.isfinite(img)
                    m,s = np.median(img[keep]),np.std(img[keep])
                    plot_clim = (m-self.clim_nstds*s,
                                 m+self.clim_nstds*s)
                plt.clim(*plot_clim)

            # Find out the transformation, and update the valid data range
            self.image_synchronizers[key] = ImageSynchronizer(ref_image, distorted_image, method=method)
            sync_mask = self.image_synchronizers[key].get_valid_mask(distorted_image)
            self.valid_mask_processed *= sync_mask

            # Apply the shift to all channels
            for channel in self.channels:
                image_distorted = self.images_processed[direction][channel][key]
                image_synchronized =self.image_synchronizers[key](image_distorted)
                self.images_processed[direction][channel][key] = image_synchronized

            if plot:
                plt.subplot(122)
                img = self.images_processed[direction][ref_channel][key] -ref_image
                (img).plot(colorbar=False)
                plt.gca().set_aspect('equal')
                plt.title('Diff after synchronizing: %s' % str((direction, ref_channel, key)))
                plt.grid(ls='--', color='w')
                if plot_clim is None:
                    keep = self.valid_mask_processed.astype(bool)*np.isfinite(img)
                    m, s = np.median(img[keep]), np.std(img[keep])
                    plot_clim = (m - self.clim_nstds * s,
                                 m + self.clim_nstds * s)
                plt.clim(*plot_clim)

    def synchronize_images_to_direction(self, ref_direction='Forward',
                                        ref_channel='S3',
                                        sync_channels=['S3','P3'],
                                        method='homography',
                                        use_valid_data=True,
                                        plot=False,plot_clim=None):

        import copy

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        # When synchronizing, perhaps we want to consider only data in the valid space
        if use_valid_data: sanitize_where = ~(self.valid_mask_processed.astype(bool))
        else: sanitize_where=None

        if self.debug: print('Synchronizing %s images to channel %s, direction %s!'%(sync_channels,ref_channel,ref_direction))
        directions_to_sync = copy.copy(self.directions)
        directions_to_sync.remove(ref_direction)

        # Loop on keys
        self.directional_synchronizers = {}
        for key in self.keys:
            # Get our reference image for this key, pointing in `ref_direction`
            ref_image = self.images_processed[ref_direction][ref_channel][key]
            ref_image = self.sanitized_image(ref_image,sanitize_where=sanitize_where)

            for direction in directions_to_sync:
                if self.debug: print('Synchronizing direction %s, key %s.'%(direction,key))
                distorted_image = self.images_processed[direction][ref_channel][key]
                distorted_image = self.sanitized_image(distorted_image,sanitize_where=sanitize_where)

                # Find out the shift
                self.directional_synchronizers[key] = ImageSynchronizer(ref_image, distorted_image, method=method)
                sync_mask = self.directional_synchronizers[key].get_valid_mask(distorted_image)
                self.valid_mask_processed *= sync_mask

        # Now actually perform the shifting using the median shifts
        for key in self.keys:
            ref_image = self.images_processed[ref_direction][ref_channel][key]
            ref_image_to_plot = self.sanitized_image(ref_image,sanitize_where=sanitize_where)

            for direction in directions_to_sync:

                # Now sync all the channels
                for channel in sync_channels:
                    distorted_image = self.images_processed[direction][channel][key]

                    if plot and channel is ref_channel:
                        distorted_image_to_plot = self.sanitized_image(distorted_image,sanitize_where=sanitize_where)
                        plt.figure(figsize=figsize)
                        plt.subplot(121)
                        img = distorted_image_to_plot-ref_image_to_plot
                        (img).plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('Diff before synchronizing: %s' % str((direction, ref_channel, key)))
                        plt.grid(ls='--', color='w')
                        if plot_clim is None:
                            keep=self.valid_mask_processed.astype(bool)*np.isfinite(img)
                            m,s = np.median(img[keep]),np.std(img[keep])
                            plot_clim = (m-self.clim_nstds*s,
                                         m+self.clim_nstds*s)
                        plt.clim(*plot_clim)

                    image_synchronized = self.directional_synchronizers[key](distorted_image)
                    self.images_processed[direction][channel][key] = image_synchronized

                    if plot and channel is ref_channel:
                        image_synchronized_to_plot = self.sanitized_image(image_synchronized,sanitize_where=sanitize_where)
                        plt.subplot(122)
                        img = image_synchronized_to_plot - ref_image_to_plot
                        (img).plot(colorbar=False)
                        plt.gca().set_aspect('equal')
                        plt.title('After synchronizing: %s' % str((direction, ref_channel, key)))
                        plt.grid(ls='--', color='w')
                        if plot_clim is None:
                            keep=self.valid_mask_processed.astype(bool)*np.isfinite(img)
                            m,s = np.median(img[keep]),np.std(img[keep])
                            plot_clim = (m-self.clim_nstds*s,
                                         m+self.clim_nstds*s)
                        plt.clim(*plot_clim)

    def get_data_cube(self,channel='S3',direction='Forward',valid_data=False,return_mask=False):

        imgs = np.array([self.images[direction][channel][key] for key in self.keys])
        x,y = self.xy

        if valid_data: #Trim to smallest rectangle with valid data
            self.update_valid_mask(use_channel=channel)

            # Replace internal missing data with medians
            mask = self.valid_mask.copy()
            for img in imgs:
                m = np.median(img[self.valid_mask])

                img[~self.valid_mask] = m

            xrange,yrange = self.get_valid_xy_range()
            imgs = imgs[:,xrange]
            imgs = imgs[:,:,yrange]
            x=x[xrange]
            y=y[yrange]

            mask = mask[xrange]
            mask = mask[:,yrange]

        else:
            mask = np.zeros(imgs[0].shape)+1

        imgs = AWA(imgs,axes=[self.keys,x,y],
                   axis_names=['Key value','X','Y'])

        self.data_cube = imgs
        self.data_cube_mask = mask

        if return_mask: return imgs, mask
        else: return imgs

    def normalized_data_cube(self,data_cube,norm_mask,
                             valid_data_mask = None,poly_order=1,
                             plot_slice=None):

        # Check that normalization mask has appropriate shape for data cube
        assert isinstance(data_cube,AWA)
        spatial_shape = data_cube.shape[1:] # First axis should be 'keys' dimension
        assert norm_mask.shape == spatial_shape

        # Check if provided data cube mask has right shape
        if valid_data_mask is not None:
            assert valid_data_mask.shape == spatial_shape
            valid_data_mask = valid_data_mask.astype(bool)
        # Or default to default mask if possible
        else:
            print('No `valid_data_mask` was provided!')
            if spatial_shape == self.data_cube_mask.shape:
                print('Defaulting to `self.data_cube_mask`...')
                valid_data_mask = self.data_cube_mask
            else:
                print('Warning: Using no `valid_data_mask`!')
                valid_data_mask=True
        norm_mask = norm_mask * valid_data_mask

        from common import numerical_recipes as numrec
        data_cube_normalized = data_cube.copy()

        for i in range(len(data_cube)):
            img = data_cube_normalized[i]
            surf = numrec.PolynomialFitImage(img.real, order=poly_order,
                                                  mask=norm_mask,
                                                  full_output=False)
            if data_cube.dtype in cmplx_dtypes:
                surf_imag = numrec.PolynomialFitImage(img.imag, order=poly_order,
                                                      mask=norm_mask,
                                                      full_output=False)
                surf = surf + 1j*surf_imag
            img /= surf

        if plot_slice is not None:
            assert isinstance(plot_slice,int)
            img = data_cube[plot_slice].copy()
            img_norm = data_cube_normalized[plot_slice].copy()

            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            np.abs(img).plot()
            plt.gca().set_aspect('equal')
            plt.title('Before normalization')

            plt.subplot(122)
            np.abs(img_norm).plot()
            plt.gca().set_aspect('equal')
            plt.title('After normalization')
            plt.tight_layout()

            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            img[~norm_mask] = np.nan
            np.abs(img).plot()
            plt.gca().set_aspect('equal')
            plt.title('Norm. region before normalization')

            plt.subplot(122)
            img_norm[~norm_mask] = np.nan
            np.abs(img_norm).plot()
            plt.gca().set_aspect('equal')
            plt.title('Norm. region after normalization')
            plt.tight_layout()

        return data_cube_normalized

class PCA_plus_GMM:

    @staticmethod
    def run_PCA(data_cube, plot=True, scale_std = False, rotate_complex=True):

        assert data_cube.ndim == 3 and data_cube.shape[0] > 2
        data_cube = data_cube.copy()

        print('Identifying principal components of data cube with shape %s...' % str(data_cube.shape))

        # Assume that frequencies are first axis
        Nfreq = data_cube.shape[0]
        spatial_shape = list(data_cube.shape[1:])
        spatial_axes = data_cube.axes[1:]
        X = data_cube.reshape((Nfreq, np.prod(spatial_shape)))  # each image pixel is a row vector
        X = X.T  # each image pixel is a column vector

        # Prepare input to covariance calculation
        Xmean = np.mean(X, axis=0)  # This will be complex
        B = X - Xmean[np.newaxis, :]  # broadcast over all spatial points
        # Disable if we want to de-emphasize noisy points?
        if scale_std:
            Xstd = np.std(X, axis=1)
            B /= Xstd[:,np.newaxis]

        # Get covariance matrix and diagonalize
        n = B.shape[0]
        C = 1 / (n - 1) * B.conj().T @ B # Hermitian covariance matrix
        from scipy.linalg import eigh
        eigvals, V = eigh(C)

        # sort with largest eigvals first
        eigvals = eigvals[::-1]
        Npcs = len(eigvals)
        principal_components = W = V[:, ::-1] # re-sort the columns

        # Plot eigenvalues
        if plot:
            plt.figure()
            plt.semilogy(eigvals, marker='o')
            plt.xlim(0, 10)
            plt.title('Principal component eigenvalues')
            plt.xlabel(r'$\nu$')

        # Compute scores
        scores = B @ np.conj(W)  # dot product the W column vectors into our data
        score_images = scores.reshape(list(spatial_shape) + [Npcs])
        score_images = score_images.transpose((2, 0, 1))
        score_images = AWA(score_images,
                           axis_names=[r'$nu$', 'X', 'Y'],
                           axes=[None] + spatial_axes)

        # If complex-valued PCA, make best choice of phase
        if score_images.dtype in cmplx_dtypes and rotate_complex:
            for i in range(Npcs):
                pc = principal_components[:,i]
                score_image = score_images[i]

                xvals = np.array(score_image).flatten()

                Nangles=200; pow=2
                angles = np.linspace(0, np.pi, Nangles)
                Xbig = xvals[:, np.newaxis] * np.exp(-1j * angles)[np.newaxis, :]
                metric = np.sum(np.abs(Xbig.real)**pow,axis=0)
                theta = angles[np.argmax(metric)]

                """
                Xvals = np.abs(xvals)

                redundant = (xvals.real == np.median(xvals.real))
                xvals = xvals[~redundant]
                Xvals = Xvals[~redundant]

                pow=1
                A=np.sum( (Xvals**pow * np.abs(xvals.imag)) )
                B=np.sum( (Xvals**pow * np.abs(xvals.real)) )
                theta = np.arctan(A/B) # SO, scores are on average rotated by `theta`
                """

                print('Rotated principal component $nu=%i$ by $theta=%1.2f$...'%(i,theta))
                phase_factor = np.exp(-1j*theta)

                score_image *= phase_factor # remove the angle from scores
                pc /= phase_factor # add the angle to principal component

        # Plot score images
        if plot:
            for i in range(Npcs):
                img = score_images[i]

                if img.dtype in cmplx_dtypes:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(121)
                    img.real.plot()
                    plt.gca().set_aspect('equal')
                    plt.title(r'Re[$\nu=%i$]' % i)

                    plt.subplot(122)
                    img.imag.plot()
                    plt.gca().set_aspect('equal')
                    plt.title(r'Im[$\nu=%i$]' % i)

                else:
                    plt.figure(figsize=(6,4))
                    img.plot()
                    plt.gca().set_aspect('equal')
                    plt.title(r'Scores for principal component $\nu=%i$' % i)

        average_component = Xmean

        return score_images,principal_components,eigvals,average_component

    def __init__(self, data_cube, valid_mask = None, weights=None, plot_PCA=True,**kwargs):
        """The only thing this will do is store the data cube, run PCA, and store the PCA results."""

        # Replace regions of invalid data with the median value
        data_cube = data_cube.copy()
        if valid_mask is not None:
            assert valid_mask.shape == data_cube.shape[1:]
            for img in data_cube: img[~valid_mask] = np.median(img[valid_mask])
        self.data_cube = data_cube

        # apply weights
        faxis = self.data_cube.axes[0]
        if weights is not None:
            assert len(weights) == len(faxis)
            weights = np.array(weights)[:, np.newaxis, np.newaxis]
        else:
            weights = np.ones((len(faxis), 1, 1))

        # --- PCA part
        data_for_PCA = self.data_cube * weights
        self.scores_cube, self.principal_components, self.eigvals,self.average_component = self.run_PCA(data_for_PCA,
                                                                                                        plot=plot_PCA,**kwargs)

    def init_clustering(self, principal_components=None,
                        nclusters=3, cluster_centers=None,
                        cluster_dimensions=None,
                        histogram_dim1=0,histogram_dim2=1,
                        nstds_binning=4,nstds_clim=.5,Nbins=100,
                        picker_radius=0.1):
        # The result of PCA, `self.scores_cube`, might have many principal components.
        # But select fewer "active" components for clustering by selecting among them with `principal_components`.
        # If scores are complex, there will be a dimensional space for clustering twice the number of "active" principal components.

        # --- Find out what which active principal components are desired for clustering
        if principal_components is None:
            principal_components = len(self.scores_cube) # size of score cube is a good default for how many principal components
        if not hasattr(principal_components, '__len__'):
            principal_components = np.arange(principal_components)  # a list of numbers
        self.active_principal_components = principal_components

        if np.any(cluster_centers):
            assert hasattr(cluster_centers,'__len__')
            self.init_cluster_centers = cluster_centers
            self.init_nclusters = len(cluster_centers)
        else:
            self.init_cluster_centers = []
            self.init_nclusters=nclusters

        # Remove any existing gmm, since we are (re-)initializing
        if hasattr(self,'gmm'):
            print('Erasing pre-existing GMM...')
            del self.gmm

        # --- Build vectors that identify points to be clustered;
        # take these from the scores of active principal components
        # If scores (and principal components) are complex, take real and imaginary parts as separate dimensions
        if self.scores_cube[0].dtype in cmplx_dtypes:
            pca_score_vecs_r = [self.scores_cube[nu].real.flatten() for \
                                nu in self.active_principal_components]
            pca_score_vecs_i = [self.scores_cube[nu].imag.flatten() for \
                                nu in self.active_principal_components]
            self.clustering_vecs = []
            # inter-leave the real and imaginary scores as separate dimensions for clustering
            for nu in range(len(self.active_principal_components)):
                self.clustering_vecs.append(pca_score_vecs_r[nu])
                self.clustering_vecs.append(pca_score_vecs_i[nu])
        else:
            self.clustering_vecs = [self.scores_cube[nu].flatten() for \
                                    nu in self.active_principal_components]  # Flatten raw images into 1D score arrays
        # @TODO: Here is where we weight by PCA importance

        # We can toss out useless or redundant dimensions.
        # For example, to use complex PCs [0,1] but cluster with (PC1.real, PC2.real, PC3.real), then we "keep dimensions" (1,2,3)
        if cluster_dimensions is not None:
            self.clustering_vecs = [self.clustering_vecs[dim] for dim in cluster_dimensions]

        self.clustering_vecs = np.array(self.clustering_vecs).T  # Put points index first
        self.clustering_ndim = self.clustering_vecs.shape[1]  # could be twice the number of principal components if scores were complex

        # Inspect the distribution of the space for clustering
        self.clustering_vec_means = [np.mean(x) for x in self.clustering_vecs.T]
        self.clustering_vec_stds = [np.std(x) for x in self.clustering_vecs.T]
        self.clustering_vec_bins = [np.linspace(x0 - nstds_binning * std,
                                                x0 + nstds_binning * std, Nbins) \
                                    for x0, std in zip(self.clustering_vec_means, self.clustering_vec_stds)]
        self.nstds_clim = nstds_clim

        # We specified how many clusters we want, so leave for GMM to figure out centers
        if self.init_nclusters is not None: return

        # Otherwise, this is our invocation to manually identify cluster centers

        #--- Now we begin the interactive part of initialization ---
        print('Beginning manual selection of cluster centers!')

        # Optionally enclose the following in loop over dimension pairs
        self.histogram_dim1 = histogram_dim1
        self.histogram_dim2 = histogram_dim2

        # Get score maps for involved principal components
        img = self.scores_cube[0]
        self.map1 = AWA(self.clustering_vecs[:, self.histogram_dim1].reshape(img.shape),
                        adopt_axes_from=img)
        self.map2 = AWA(self.clustering_vecs[:, self.histogram_dim2].reshape(img.shape),
                        adopt_axes_from=img)

        # Get coordinates in latent variable space and also real-space

        self.latent_coords_all = list(zip(self.clustering_vecs[:, self.histogram_dim1],
                                          self.clustering_vecs[:, self.histogram_dim2]))
        self.latent_coords_all = np.array(self.latent_coords_all)
        self.map_X,self.map_Y =self.map1.axis_grids
        self.xy_coords_all = np.array(list(zip(self.map_X.flatten(),
                                               self.map_Y.flatten())))

        # Workflow
        # 1) plot histogram
        # 2) Initiate lasso
        # 3) Lasso is active on each selection, will do `onLassoSelect`
        # 4) Enter key allows us to proceed after all clusters are selected
        self.fig = plt.figure(figsize=(12,6))

        # Plot the scores for one of the involved principal components
        self.ax_map = plt.subplot(121)
        self.map1.plot(colorbar=False,cmap='plasma')
        plt.gca().set_aspect('equal')
        #c0 = np.mean(map1); std = 2*np.std(map1)
        #plt.clim(c0-self.nstds_clim*std,c0+self.nstds_clim*std)
        plt.tight_layout()
        plt.title(r'Scores for latent dimension %i' % self.histogram_dim1)

        # Plot the histogram in these dimensions
        self.ax_histogram = plt.subplot(122)
        self.histogram2D(self.histogram_dim1,
                         self.histogram_dim2,
                         ax=self.ax_histogram)
        self.ax_histogram_default_title = 'Select with lasso; hit enter if to keep, escape to finish!'
        self.ax_histogram.set_title(self.ax_histogram_default_title)

        from matplotlib.widgets import LassoSelector
        events = []
        events.append( LassoSelector(self.ax_map, onselect=self.onLassoMap) )
        events.append( LassoSelector(self.ax_histogram, onselect=self.onLassoHistogram) )
        events.append( self.fig.canvas.mpl_connect("key_press_event", self.accept_center) )
        events.append( self.fig.canvas.mpl_connect("key_press_event", self.done_selecting_centers) )
        events.append( self.fig.canvas.mpl_connect("button_press_event", self.click_map) )
        self.picker_radius = picker_radius
        self.events=events
        self.draw()

        print('Run `run_GMM()` once you have selected the initial cluster centers!')

    def histogram1D(self,dim=0):

        bins = self.clustering_vec_bins[dim]
        scores = self.clustering_vecs[:, dim]

        h = num.bin_array(scores,bins=bins)
        h.plot()
        #h=plt.hist(scores.flatten(), bins=bins, edgecolor='black')
        plt.xlabel('PC %i' % dim)
        plt.ylabel('Frequency')
        plt.tight_layout()

        return h

    def histogram2D(self, dim1, dim2,
                    rot_angle=0,ax=None,**kwargs):

        from matplotlib.colors import LogNorm

        bins1, bins2 = self.clustering_vec_bins[dim1], self.clustering_vec_bins[dim2]
        scores1, scores2 = self.clustering_vecs[:, dim1], self.clustering_vecs[:, dim2]
        X = np.cos(rot_angle)*scores1 + np.sin(rot_angle)*scores2
        Y = -np.sin(rot_angle)*scores1 + np.cos(rot_angle)*scores2

        if 'norm' not in kwargs:
            kwargs['norm'] = LogNorm(vmin=0.01, vmax=1)

        if ax: plt.sca(ax)
        h = plt.hist2d(X,Y,
                       bins=(bins1, bins2),
                       **kwargs)
        #plt.gca().set_aspect('equal')

        cmax = np.std(h[0]) * self.nstds_clim
        plt.clim(cmax / 100, cmax)

        plt.xlabel('PC %i' % dim1)
        plt.ylabel('PC %i' % dim2)
        plt.title('PCs %i & %i' % (dim1, dim2))
        plt.tight_layout()

        return h

    def draw(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.show(block=False)

    def onLassoMap(self,verts):

        from matplotlib.path import Path

        path = Path(verts)
        included_inds = np.nonzero(path.contains_points(self.xy_coords_all))[0]
        included_scores1, included_scores2 = list(zip(*self.latent_coords_all[included_inds]))
        mean_score1, mean_score2 = np.mean(included_scores1), np.mean(included_scores2)

        plt.sca(self.ax_histogram)
        try: self.tentative_xy_marker.remove() # Remove any former marker
        except: pass
        self.tentative_xy = (mean_score1, mean_score2)
        self.tentative_xy_marker,=plt.plot([mean_score1],
                                           [mean_score2],
                                           marker='o',markersize=10,
                                           markeredgecolor='w')
        self.draw()

    def onLassoHistogram(self, verts):
        from matplotlib.path import Path

        path = Path(verts)
        included_inds = np.nonzero(path.contains_points(self.latent_coords_all))[0]
        included_scores1, included_scores2 = list(zip(*self.latent_coords_all[included_inds]))
        mean_score1, mean_score2 = np.mean(included_scores1), np.mean(included_scores2)
        self.tentative_xy = (mean_score1,mean_score2)

        # Pinpoint the mean score
        plt.sca(self.ax_histogram)
        try: self.tentative_xy_marker.remove() # Remove any former marker
        except: pass
        self.tentative_xy_marker,=plt.plot([mean_score1],
                                           [mean_score2],
                                           marker='o',markersize=10,
                                           markeredgecolor='w',
                                           zorder=10)

        img = self.scores_cube[0]
        overlay_map = AWA(np.full(img.shape,np.nan),
                          adopt_axes_from=img)
        overlay_map[np.unravel_index(included_inds,
                                     img.shape)]=1 # finite only at included pixels

        plt.sca(self.ax_map)
        try: self.overlay_im.remove() # Remove any former overlay
        except: pass
        self.overlay_im = overlay_map.plot(alpha=1,colorbar=False) # Add our new overlay
        plt.gca().set_aspect('equal')

        self.draw()

        self.lasso_calls +=1

    def click_map(self,event):

        if event.button == 1:
            x,y =event.xdata, event.ydata
            mean_score1,mean_score2 = self.map1.cslice[x,y],self.map2.cslice[x,y]
            self.tentative_xy = (mean_score1,mean_score2)
            #picked_region = ( (x-self.map_X)**2 + (y-self.map_Y)**2  < self.picker_radius**2 )
            #mean_score1,mean_score2 = np.median(self.map1[picked_region]),\
            #                          np.median(self.map2[picked_region])
            #self.tentative_xy = (mean_score1,mean_score2)
            plt.sca(self.ax_map)
            try: self.tentative_map_point.remove()
            except: pass
            self.tentative_map_point,=plt.plot([x],[y],marker='o',markersize=10,markeredgecolor='w',
                                               zorder=10)

            # Pinpoint the mean score
            plt.sca(self.ax_histogram)
            try: self.tentative_xy_marker.remove() # Remove any former marker
            except: pass
            self.tentative_xy_marker,=plt.plot([mean_score1],
                                               [mean_score2],
                                               marker='o',markersize=10,
                                               markeredgecolor='w',
                                               zorder=10)

    def accept_center(self, event):

        if event.key == "enter":

            new_cluster_center = [0 for i in range(self.clustering_ndim)] # Initialize an empty center
            new_cluster_center[self.histogram_dim1] = self.tentative_xy[0]
            new_cluster_center[self.histogram_dim2] = self.tentative_xy[1]
            self.init_cluster_centers.append(new_cluster_center)
            self.init_nclusters = len(self.init_cluster_centers) # Since we don't have a gmm yet, update the intended number of clusters

            try: self.tentative_xy_marker.remove()
            except: pass
            cluster_no = len(self.init_cluster_centers)

            text_fontsize=12

            # Label the cluster center on histogram
            plt.sca(self.ax_histogram)
            t=plt.text(self.tentative_xy[0], self.tentative_xy[1],cluster_no,
                       color='w',fontsize=text_fontsize+2,weight='bold',
                       va="center",ha="center")
            t=plt.text(self.tentative_xy[0], self.tentative_xy[1],cluster_no,
                       color='k',fontsize=text_fontsize,
                       va="center",ha="center")

            # Label the cluster center in real-space
            plt.sca(self.ax_map)
            pos_ind = np.argmin((self.map1-self.tentative_xy[0])**2
                                  + (self.map2-self.tentative_xy[1])**2)
            pos_xy = self.xy_coords_all[pos_ind]
            t=plt.text(pos_xy[0], pos_xy[1],cluster_no,
                       color='w',fontsize=text_fontsize+2,weight='bold',
                       va="center",ha="center")
            t=plt.text(pos_xy[0], pos_xy[1],cluster_no,
                       color='k',fontsize=text_fontsize,
                       va="center",ha="center")
            #t.set_bbox(dict(alpha=text_alpha,color='w'))
            #self.ax_histogram.set_title("Selected cluster %i!" % cluster_no)

            #time.sleep(1)  # Give a message, then restore original title
            self.ax_histogram.set_title(self.ax_histogram_default_title)

            self.draw()

    def done_selecting_centers(self,event):
        if event.key == "escape":
            for event in self.events: self.fig.canvas.mpl_disconnect(event)
            self.ax_histogram.set_title("Successfully selected %i cluster centers!" % self.get_gmm_nclusters())

            self.draw()

    def run_GMM(self, random_state=0, tol=1e-9, max_iter=1000,
                init_params='k-means++', **kwargs):

        from sklearn.mixture import GaussianMixture
        self.random_state = random_state
        ndim = np.array(self.clustering_vecs).shape[1]

        # `warm_start = True` allows repeat calls to `.fit` to skip re-initialization
        t0=time.time()
        if len(self.init_cluster_centers):
            print('Running GMM with manually identified %i cluster centers on %i-dimensional data...' % (self.get_gmm_nclusters(), ndim))
            self.gmm = GaussianMixture(n_components=self.get_gmm_nclusters(),
                                       means_init=self.init_cluster_centers, tol=tol, max_iter=max_iter, warm_start=True,
                                       **kwargs).fit(self.clustering_vecs)
        else:
            print('Running GMM with %s initialization with %i clusters on %i-dimensional data...' % (init_params, self.get_gmm_nclusters(), ndim))
            self.gmm = GaussianMixture(n_components=self.get_gmm_nclusters(),
                                       init_params=init_params, tol=tol, max_iter=max_iter,
                                       random_state=random_state, warm_start=True,
                                       **kwargs).fit(self.clustering_vecs)
        print('Time elapsed: %1.2f seconds'%(time.time()-t0))

        self.init_cluster_centers = self.gmm.means_

    def rerun_GMM(self,max_iter=None):

        if max_iter: self.gmm.max_iter=max_iter
        ndim = np.array(self.clustering_vecs).shape[1]
        print('Re-running GMM with %i clusters on %i-dimensional data...' % (self.get_gmm_nclusters(), ndim))

        t0=time.time()
        result = self.gmm.fit(self.clustering_vecs)
        print('Time elapsed: %1.2f seconds'%(time.time()-t0))

        return result

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse(position, width, height,
                             angle=angle, **kwargs))

    def visualize_gmm_clusters(self,dim1=0,dim2=1,
                               text_fontsize=12,
                               label_reordering=None,
                               **kwargs):

        self.fig=plt.figure(figsize=(8,6))
        self.ax_histogram=plt.subplot(111)

        ndim = self.gmm.means_.shape[1]
        if ndim == 1:
            self.histogram1D(dim=0)
            plt.twinx()
            bins = self.clustering_vec_bins[0]
            for pos,covar,w in zip(self.gmm.means_[:,0],
                                   self.gmm.covariances_[:,0],
                                   self.gmm.weights_):
                g = w*np.exp( -(bins-pos)**2/(2*covar) )
                plt.fill_between(bins,0,g,color='b',edgecolor='b',alpha=.4)
            return

        # Make a new histogram plot
        self.histogram2D(dim1, dim2, ax=self.ax_histogram,**kwargs)

        # Loop over the identified clusters
        w_factor = 100 / self.gmm.weights_.max()
        cluster_no=0
        for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            # Find projection of this cluster onto dim1,dim2
            pos_dim12 = (pos[dim1], pos[dim2])
            covar_dim12 = np.array([[covar[dim1, dim1], covar[dim1, dim2]],
                                    [covar[dim2, dim1], covar[dim2, dim2]]])
            alpha = w * w_factor
            self.draw_ellipse(pos_dim12, covar_dim12, alpha=.5, ax=self.ax_histogram,
                              edgecolor='w')

            if label_reordering is not None:
                try: cluster_label = list(label_reordering).index(cluster_no)
                except ValueError:
                    raise ValueError('cluster number %i must be found within `label reordering` list!'%cluster_no)
            else:
                cluster_label = cluster_no

            # Label with a text number
            t=plt.text(pos_dim12[0], pos_dim12[1],cluster_label,
                       color='w',fontsize=text_fontsize+2,weight='bold',
                       va="center",ha="center")
            t=plt.text(pos_dim12[0], pos_dim12[1],cluster_label,
                       color='k',fontsize=text_fontsize,
                       va="center",ha="center")
            cluster_no+=1

        #self.draw()

    @staticmethod
    def smooth_segmentation(segmentation, smoothing=1, method='gaussian'):

        import cv2
        segmentation = segmentation.copy().astype(float)
        seg_vals = np.unique(segmentation)

        for seg_val in seg_vals:
            mask = (segmentation == seg_val).astype(float)
            # Since we fiddle with this region of the segmentation, its values are "unknown" here
            segmentation[mask.astype(bool)] = np.nan

            if method == 'gaussian':
                new_mask = cv2.GaussianBlur(mask, (0,) * 2, smoothing)  # kx,ky will be figured out by smoothing
            else: # use erosion+dilation
                kernel = np.ones((int(np.round(smoothing)) + 1,) * 2)
                new_mask = cv2.erode(mask, kernel, iterations=1)
                new_mask = cv2.dilate(new_mask, kernel, iterations=1)

            new_mask = np.round(new_mask).astype(bool)
            segmentation[new_mask] = seg_val

        # Fill in the unknown `nan` data with that from nearest pixels
        # Recipe from https://stackoverflow.com/questions/68197762/fill-nan-with-nearest-neighbor-in-numpy-array
        from scipy.interpolate import NearestNDInterpolator
        valid = np.where(~np.isnan(segmentation))
        interp = NearestNDInterpolator(np.transpose(valid), segmentation[valid])
        segmentation = interp(*np.indices(segmentation.shape))

        return segmentation.astype(int)

    def get_gmm_nclusters(self):

        try: return len(self.gmm.means_)
        except AttributeError: # Maybe we didn't run GMM yet
            return self.init_nclusters

    def get_gmm_centers(self):

        try:
            return copy.copy(self.gmm.means_)
        except AttributeError:  # Maybe we didn't run GMM yet
            return self.init_cluster_centers

    def get_gmm_maps(self, smoothing=0.7,order_by = 'amplitude',
                           plot=True,cmap='viridis',**kwargs):

        assert order_by in ['amplitude','population',None]
        # Predict the labels
        print('Predicting cluster membership...')
        labels = self.gmm.predict(self.clustering_vecs)
        # reshape labels
        img = self.scores_cube[0]
        labels_map = np.reshape(labels, img.shape)

        #Smooth labels
        if smoothing:
            labels_map = self.smooth_segmentation(labels_map,smoothing,**kwargs)

        # reorder labels
        old_labels = np.unique(labels_map).astype(int)
        if order_by =='population':
            reordering_target = [np.sum(labels_map==label) for label in old_labels]
        elif order_by == 'amplitude':
            mean_amp_img = np.sum(np.abs(self.data_cube),axis=0) # sum on frequency axis
            reordering_target = [np.mean(mean_amp_img[labels_map==label]) for label in old_labels]
        else: reordering_target = np.arange(len(old_labels))
        reordering = np.argsort(reordering_target) # smallest label will be first

        new_labels_map = labels_map.copy()
        for new_label,old_label in enumerate(reordering):
            new_labels_map[labels_map == old_label] = new_label
        labels_map = new_labels_map

        self.labels_map = AWA(labels_map, adopt_axes_from=img)

        if plot:
            self.fig = plt.figure()
            self.ax_map = plt.gca()
            self.labels_map.plot(cmap=cmap)
            plt.gca().set_aspect('equal')

        maps = {}
        maps['labels']=self.labels_map

        # Get probabilities of belonging to each cluster
        probs = self.gmm.predict_proba(self.clustering_vecs)

        self.prob_maps = [AWA(np.reshape(prob, img.shape),
                          adopt_axes_from=img)
                      for prob in probs.T[reordering]]

        maps['probabilities']=self.prob_maps
        maps['cluster_ids']=old_labels[reordering]
        maps['label_reordering'] = reordering
        self.maps = maps

        return maps

    def get_gmm_populations(self,order_by='population',**kwargs):

        labels_map = self.get_gmm_maps(plot=False, order_by=order_by,
                                       smoothing=0,**kwargs)['labels']
        labels = np.unique(labels_map.astype(int))
        pops = np.array([float(np.sum(labels_map == label)) for label in labels])

        return AWA(pops,axes=[labels],axis_names=[r'Label $\nu$'])

    def get_gmm_fidelity(self, smoothing=0.6, **kwargs):

        # How well do we reproduce the original data?
        labels_map = self.get_gmm_maps(smoothing=smoothing, plot=False, **kwargs)['labels']
        data_cube = self.data_cube
        dummy_data_cube = np.zeros(data_cube.shape,dtype=complex)
        Nkeys = dummy_data_cube.shape[0]
        for label in np.unique(labels_map):
            mask = (label == labels_map)
            for i in range(Nkeys):
                dummy_data_cube[i][mask] = np.mean(data_cube[i][mask])

        diff = np.sum(np.abs(dummy_data_cube - data_cube) ** 2)
        norm = np.sum(np.abs(data_cube) ** 2)
        metric = 1 - diff / norm

        return float(metric)

    get_gmm_quality = get_gmm_fidelity # An alias

    def get_gmm_connectivities(self,order_by='population',thresh=None):
        # This will tell us how domain "i" is connected to domain "j", relative to its self-connection

        maps = self.get_gmm_maps(plot=False, order_by=order_by,smoothing=0)
        N = len(maps['probabilities'])
        connectivities = np.zeros((N, N))
        for i in range(N):
            map_i = maps['probabilities'][i]
            Norm = np.sum(map_i ** 2)
            for j in range(N):
                if i==j: continue
                map_j = maps['probabilities'][j]
                connectivities[i, j] = np.sum(map_i * map_j) / Norm

        # Specify a threshold to focus only on those connections exceeding a threshold strength (others suppressed)
        if thresh: connectivities[connectivities<thresh] = 0

        return connectivities

    def get_gmm_connectivity(self):

        order_by = 'population'
        Cs = self.get_gmm_connectivities(order_by=order_by)
        C = np.sum(Cs,axis=1)
        populations = np.array(self.get_gmm_populations(order_by=order_by))
        Cavg =np.sum(C*populations)/np.sum(populations)

        return Cavg

    def prune_gmm_clusters(self,cluster_ids,run_gmm=False,**kwargs):

        # Edit the gmm in-situ by removing
        self.gmm.means_bak_ = self.gmm.means_
        self.gmm.covariances_bak_ = self.gmm.covariances_
        self.gmm.weights_bak_ = self.gmm.weights_
        self.gmm.precisions_cholesky_bak_ = self.gmm.precisions_cholesky_

        keep = np.array([True]*len(self.gmm.means_bak_))
        for cluster_id in cluster_ids: keep[cluster_id]=False

        print('Pruning from %i clusters down to %i clusters.' % (self.get_gmm_nclusters(), np.sum(keep)))

        self.init_cluster_centers = self.gmm.means_ = self.gmm.means_[keep]
        self.gmm.covariances_ = self.gmm.covariances_[keep]
        self.gmm.weights_ = self.gmm.weights_[keep]
        self.gmm.precisions_cholesky_ = self.gmm.precisions_cholesky_[keep]

        if run_gmm: self.run_GMM(**kwargs)

    def prune_gmm_to_target_nclusters(self,target_nclusters=12,
                                      rerun_GMM=False,**kwargs):

        # Lop off one cluster at a time until we hit desired number
        current_nclusters = self.get_gmm_nclusters()
        while current_nclusters > target_nclusters:
            Cs = self.get_gmm_connectivities()
            C = np.sum(Cs,axis=1)
            cluster_id_to_prune = self.maps['cluster_ids'][np.argmax(C)]
            self.prune_gmm_clusters([cluster_id_to_prune])
            current_nclusters = self.get_gmm_nclusters()
            if rerun_GMM: self.rerun_GMM(**kwargs)

    def prune_gmm_to_target_connectivity(self,target_connectivity=0.5,
                                         rerun_GMM=False,**kwargs):

        # Lop off one cluster at a time until we hit desired number
        Cs = self.get_gmm_connectivities()
        C = np.sum(Cs, axis=1)
        Cmax = np.max(C)
        while Cmax > target_connectivity:
            cluster_id_to_prune = self.maps['cluster_ids'][np.argmax(C)]
            self.prune_gmm_clusters([cluster_id_to_prune])
            if rerun_GMM: self.rerun_GMM(**kwargs)
            Cs = self.get_gmm_connectivities()
            C = np.sum(Cs,axis=1)
            Cmax = np.max(C)

    #def pruning_loop(self,n_clusters_target,connectivity_thresh=0.3):
        # This loop will eliminate minority clusters with connectivity above `connectivity_thresh`.
        # It will stop when connectivity is eliminated, or number of clusters hits `n_clusters_target`.
        # whichever comes first.




class AutoRepLinescan(Hyperspectroscopy):
    """This is a version of Hyperspectroscopy with (for now) just one key,
    but extra features to interpret the x-axis as a segmented energy axis."""

    #@TODO: include 'valid_mask' in procedures so that synchronization of foward / backward can work

    def __init__(self,directory,segment_labels,
                 channels=['ZHeight','Aux3','Aux4'],
                 directions=['Forward','Backward'],
                 transpose=False,
                 key='default'):
        # Transpose to make slow axis (energy) increment horizontally along x-axis

        filenames = os.listdir(directory)
        self.images = {}
        self.segment_labels_provided = segment_labels # These will be e.g. wavelengths or frequencies
        self.keys = [key]
        self.channels = channels
        self.directions = directions

        for direction in self.directions:
            self.images[direction]={}
            for channel in self.channels:

                filename = list(filter(lambda f: channel in f and direction in f,
                                                 filenames))[0]
                filepath = os.path.join(directory,filename)

                img = Hyperspectroscopy.load_image(filepath)
                if transpose: img=img.T
                self.images[direction][channel] = {self.keys[0]: img}

        self.valid_mask = np.full(self.images[direction][channel][self.keys[0]].shape,True)

    def equalize_xy_axes(self):

        img = self.images[self.directions[0]][self.channels[0]][self.keys[0]]
        xs,ys = img.axes
        dx,dy = np.ptp(xs),np.ptp(ys)

        if dx>dy: new_axes = [xs,np.linspace(0,dx,len(ys))]
        else: new_axes = [np.linspace(0,dy,len(xs)),ys]

        for direction in self.directions:
            for channel in self.channels:
                for key in self.keys:
                    img = self.images[direction][channel][key]
                    img.set_axes(new_axes)

    def identify_segments(self,ref_channel='Aux4',ref_direction='Forward',
                          min_width=14,max_width=16,Nwidths=10, buffer=3,
                          smoothing=1):

        from common import numerical_recipes as numrec
        ref_img = self.images[ref_direction][ref_channel][self.keys[0]]

        # assume fast axis is 1, average over that one
        gx = np.gradient(np.mean(ref_img, axis=1))
        if smoothing:
            gx = numrec.smooth(gx, window_len=smoothing)

        plt.figure()
        np.abs(gx).plot()
        plt.title('Switch spots on %s, %s'%(ref_channel,ref_direction))

        Nxs = np.linspace(min_width, max_width, Nwidths)
        dx = np.diff(gx.axes[0])[0]

        to_maximize_list = []
        for Nx in Nxs:
            to_maximize = 0
            i = ind = 0
            while ind < ref_img.shape[0]:
                to_maximize += gx[ind]
                i += 1
                ind = int(np.round(i * Nx))
            N_switches = i + 1
            to_maximize /= N_switches
            to_maximize_list.append(to_maximize)

        Nx = Nxs[np.argmax(to_maximize_list)]
        print('Best segment width:', Nx)
        i = 0
        while i * Nx < len(gx):
            plt.axvline(dx * Nx * i, ls='--', color='k', alpha=.5)
            i += 1

        plt.figure()
        plt.plot(Nxs, to_maximize_list)
        plt.ylabel('Residual')
        plt.xlabel('Segment width')
        plt.axvline(Nx, ls='--', color='k', alpha=.5)

        #--- Now that we have segment widths, we can get segment labels map
        segment_labels = []
        for i in range(ref_img.shape[0]):

            segment_ind = int(i / Nx)
            segment_progress = i % Nx
            if segment_progress < buffer / 2 or (Nx - segment_progress) < buffer / 2:
                segment_labels.append(np.nan)
            else:
                segment_labels.append(self.segment_labels_provided[segment_ind])

        # duplicate over all entries of fast axis, which we averaged over
        segment_labels_map = np.array([segment_labels for i in range(ref_img.shape[1])])

        # as it stands, fast axis is 0, but we want it to be 1, so transpose
        segment_labels_map = segment_labels_map.T

        segment_labels_map = AWA(segment_labels_map, adopt_axes_from=ref_img)

        plt.figure()
        segment_labels_map.plot()
        plt.title('Segments')

        self.segment_labels_map = segment_labels_map
        self.segment_labels = list(filter(np.isfinite,
                                          np.unique(segment_labels_map)))

        return self.segment_labels_map

    def correct_drift_with_topo_channel(self,topo_channel = 'Z Height',
                                        plot=True):

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        for direction in self.directions:
            topo = self.images[direction][topo_channel][self.keys[0]]
            ys = topo.axes[1]
            y0 = np.mean(ys)

            if plot:
                plt.figure()
                topo.plot()
                plt.title('Before drift correction: %s, %s'%(topo_channel,direction))

            for i in range(len(topo)):
                topo_line = topo[i]
                topo_line = topo_line - np.median(topo_line)
                shift = np.sum((ys - y0) * topo_line) / np.sum(topo_line)
                new_ys = ys - shift
                for channel in self.channels:
                    img = self.images_processed[direction][channel][self.keys[0]]
                    img[i] = img[i].interpolate_axis(new_ys, axis=0,
                                                     extrapolate=True, bounds_error=False)

                    if plot and channel == topo_channel:
                        plt.figure()
                        img.plot()
                        plt.title('After drift correction: %s, %s' % (topo_channel, direction))

    @staticmethod
    def farfield_flatten_residual(params, coords, target_line, zone_masks, multiplicative=False,pow=1):

        model = np.polyval(params, coords)
        vals = []
        for zone_mask in zone_masks:
            zone_model = model[zone_mask]
            zone_target = target_line[zone_mask]
            # subtract or divide depending on amp/phase
            if multiplicative:
                zone_model_rel = zone_model / np.mean(zone_model)
                zone_target_rel = zone_target / np.mean(zone_target)
            else:
                zone_model_rel = zone_model - np.mean(zone_model)
                zone_target_rel = zone_target - np.mean(zone_target)

            new_vals = np.abs(zone_model_rel - zone_target_rel)**pow / len(zone_target_rel)
            vals = np.append(vals, new_vals)

        return np.array(vals)

    def far_field_correct(self,channel,directions=None,
                          zones=[(0, 3.6), (3.65, 4.4), (4.4, 7)],
                          multiplicative=False, deg=10,
                          plot=True, nstds_clim=4,pow=1,
                          **kwargs):

        from common import plotting
        from scipy.optimize import leastsq

        self.images_processed = copy.deepcopy(self.images)
        self.valid_mask_processed = copy.deepcopy(self.valid_mask)

        linefunc = plt.axhline

        if directions is None: directions = self.directions
        for direction in directions:
            
            img = self.images_processed[direction][channel][self.keys[0]]
            img_before = img.copy()
            coords = img.axes[1]
    
            if plot:
                plt.figure()
                img.plot()
                plt.title('Before far-field correction: %s'%direction)
                m = np.median(img[np.isfinite(img)])
                s = np.std(img[np.isfinite(img)])
                plt.clim(m - nstds_clim * s, m + nstds_clim * s)
    
            # make zone masks and plot their locations
            self.zone_masks = []
            cs = plotting.bluered_colors(len(zones))
            for start, stop in zones:
                if plot:
                    c = next(cs)
                    linefunc(start, color=c)
                    linefunc(stop, color=c)
                self.zone_masks.append((coords > start) * (coords < stop))
    
            # Loop over segments
            segment_labels = filter(np.isfinite, np.unique(self.segment_labels_map))
            for segment_label in segment_labels:
                # print('Fitting segment label ',segment_label)
                segment_mask = self.segment_labels_map == segment_label
                segment_span = segment_mask[:,0] # every y-value has the same span so just pick one
                target_line = np.mean(img[segment_span,:], axis=0)
                args = (coords, target_line, self.zone_masks, multiplicative,pow)
    
                params0 = np.polyfit(coords, target_line, deg=deg)
                # (target_line-np.mean(target_line)).plot()
                # model = np.polyval(params0,ys)
                # plt.plot(ys,model - np.mean(model))
                params_fit = leastsq(self.farfield_flatten_residual, params0,
                                     args=args,**kwargs)[0]  # First element is the fit result
                self.model_ff_factor = np.polyval(params_fit, coords)

                model_ff_factor = self.model_ff_factor[np.newaxis,:] # apply uniformly across segment width along x-axis
                if multiplicative:
                    img_modified = img / model_ff_factor
                    img_modified /= np.mean(img_modified[segment_mask]) # the mean is ad hoc, so bring to 1
                else:
                    img_modified = img - model_ff_factor
                    img_modified -= np.mean(img_modified[segment_mask]) # the offset is ad hoc, so bring to zero

                # Make the update to image only within this segment
                img = np.where(segment_mask,img_modified,img)
                
                #where_compare = np.isfinite(img)
                #print('Image changed?',np.any(img_before[where_compare] != img[where_compare]))

            img = AWA(img,adopt_axes_from = img_before)
            self.images_processed[direction][channel][self.keys[0]] = img

            if plot:
                plt.figure()
                img.plot()
                plt.title('After far-field correction: %s'%direction)
                cs = plotting.bluered_colors(len(zones))
                for start, stop in zones:
                    c = next(cs)
                    linefunc(start, color=c)
                    linefunc(stop, color=c)
                m = np.median(img[np.isfinite(img)])
                s = np.std(img[np.isfinite(img)])
                plt.clim(m - nstds_clim * s, m + nstds_clim * s)

    def resample_linescan(self,target_mask,channel='Aux3',direction='Forward'):

        target_dict = self.images
        linescan_img = target_dict[direction][channel][self.keys[0]]

        resampled_mean = [np.mean(linescan_img[(self.segment_labels_map == segment_label)*target_mask]) \
                     for segment_label in self.segment_labels]
        resampled_mean = AWA(resampled_mean,axes=[self.segment_labels],
                        axis_names=['Segment name'])

        resampled_std = [np.std(linescan_img[(self.segment_labels_map == segment_label)*target_mask]) \
                        for segment_label in self.segment_labels]
        resampled_std = AWA(resampled_std,axes=[self.segment_labels],
                        axis_names=['Segment name'])

        return resampled_mean,resampled_std