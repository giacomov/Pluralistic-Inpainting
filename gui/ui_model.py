from gui.ui_window import Ui_Form
from gui.ui_draw import *
from PIL import Image, ImageQt
import random, io, os
import numpy as np
import torch
import torchvision.transforms as transforms
from util import task, util
from dataloader.image_folder import make_dataset
from model import create_model
from util.visualizer import Visualizer

from data_science_tools.d3 import ColoredMesh, NearmapDSM
import trimesh.visual


class ui_model(QtWidgets.QWidget, Ui_Form):
    shape = 'line'
    CurrentWidth = 15

    def __init__(self, opt):
        super(ui_model, self).__init__()
        self.setupUi(self)
        self.opt = opt
        # self.show_image = None
        self.show_result_flag = False
        self.opt.loadSize = [256, 256]
        self.visualizer = Visualizer(opt)
        self.model_name = ['cape_1', 'celeba_center', 'paris_center', 'imagenet_center', 'place2_center',
                           'celeba_random', 'paris_random','imagenet_random', 'place2_random']
        self.img_root = '/home/giacomov/data/hackathon_July2020/data/'
        self.img_files = ['processed', 'celeba-hq', 'paris', 'imagenet', 'place2']
        self.graphicsView_2.setMaximumSize(self.opt.loadSize[0]+30, self.opt.loadSize[1]+30)

        # show logo
        self.show_logo()

        # original mask
        self.new_painter()

        # selcet model
        self.comboBox.activated.connect(self.load_model)

        # load image
        self.pushButton.clicked.connect(self.load_image)

        # random image
        self.pushButton_2.clicked.connect(self.random_image)

        # save result
        self.pushButton_4.clicked.connect(self.save_and_display_mesh)

        # draw/erasure the mask
        self.radioButton.toggled.connect(lambda: self.draw_mask('line'))
        self.radioButton_2.toggled.connect(lambda: self.draw_mask('rectangle'))
        self.spinBox.valueChanged.connect(self.change_thickness)
        # erase
        self.pushButton_5.clicked.connect(self.clear_mask)

        # fill image, image process
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pushButton_3.clicked.connect(self.fill_mask)

        # show the result
        self.pushButton_6.clicked.connect(self.show_result)

    def showImage(self, fname):
        """Show the masked images"""
        value = self.comboBox.currentIndex()
        img = Image.open(fname).convert('RGB')
        self.img_original = img.resize(self.opt.loadSize)
        if True:
            self.img = self.img_original
        else:
            self.img = self.img_original
            sub_img = Image.fromarray(np.uint8(255*np.ones((128, 128, 3))))
            mask = Image.fromarray(np.uint8(255*np.ones((128, 128))))
            self.img.paste(sub_img, box=(64, 64), mask=mask)
        self.show_image = ImageQt.ImageQt(self.img)
        self.new_painter(self.show_image)

    def show_result(self):
        """Show the results and original image"""
        if self.show_result_flag:
            self.show_result_flag = False
            img_out = util.tensor2im(self.img_out.detach())
            new_pil_image = Image.fromarray(img_out)
            new_qt_image = ImageQt.ImageQt(new_pil_image)
        else:
            self.show_result_flag = True
            new_qt_image = ImageQt.ImageQt(self.img_original)
        self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(new_qt_image))
        self.graphicsView_2.scene.addItem(item)
        self.graphicsView_2.setScene(self.graphicsView_2.scene)

    def show_logo(self):
        """Show the logo of NTU and BTC"""
        img = QtWidgets.QLabel(self)
        img.setGeometry(650, 20, 256, 50)
        # read images
        pixmap = QtGui.QPixmap("./gui/logo/cape.png")
        pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        img.setPixmap(pixmap)
        img.show()

    def load_model(self):
        """Load different kind models for different datasets and mask types"""
        value = self.comboBox.currentIndex()
        if value == 0:
            raise NotImplementedError("Please choose a model")
        else:
            # define the model type and dataset type
            index = value-1
            self.opt.name = self.model_name[index]
            self.opt.img_file = self.img_root + self.img_files[index % len(self.img_files)]
        self.model = create_model(self.opt)

    def load_image(self):
        """Load the image"""
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, 'Image files(*.jpg *.png)')
        self.showImage(self.fname)

    def random_image(self):
        """Random load the test image"""

        # read random mask
        if self.opt.mask_file != "none":
            mask_paths, mask_size = make_dataset(self.opt.mask_file)
            item = random.randint(0, mask_size - 1)
            self.mname = mask_paths[item]

        image_paths, image_size = make_dataset(self.opt.img_file)
        item = random.randint(0, image_size-1)
        self.fname = image_paths[item]
        self.showImage(self.fname)

    def save_and_display_mesh(self):
        """Save the results to the disk"""
        util.mkdir(self.opt.results_dir)
        img_name = self.fname.split('/')[-1]
        data_name = self.opt.img_file.split('/')[-1].split('.')[0]

        # save the original image
        original_name = '%s_%s_%s' % ('original', data_name, img_name)
        original_path = os.path.join(self.opt.results_dir, original_name)
        img_original = util.tensor2im(self.img_truth)
        util.save_image(img_original, original_path)

        # save the mask
        mask_name = '%s_%s_%d_%s' % ('mask', data_name, self.PaintPanel.iteration, img_name)
        mask_path = os.path.join(self.opt.results_dir, mask_name)
        img_mask = util.tensor2im(self.img_m)
        # Set mask to black
        img_mask[img_mask == (127, 127, 127)] = 0
        util.save_image(img_mask, mask_path)

        # save the results
        result_name = '%s_%s_%d_%s' % ('result', data_name, self.PaintPanel.iteration, img_name)
        result_path = os.path.join(self.opt.results_dir, result_name)
        img_result = util.tensor2im(self.img_out)
        util.save_image(img_result, result_path)

        # Create and show 3d mesh

        # skimage.io.imsave("__input.png", img_mask)
        # skimage.io.imsave("__model_prediction.png", img_result)

        inp = NearmapDSM.read(mask_path)
        out = NearmapDSM.read(result_path)

        cm_inp = ColoredMesh.from_dem(inp, max_slope=50)
        cm_out = ColoredMesh.from_dem(out, max_slope=50)

        cm_inp.save(f"{self.opt.results_dir}/input.ply")
        cm_out.save(f"{self.opt.results_dir}/model_prediction.ply")

        for facet in cm_out.mesh.facets:
            cm_out.mesh.visual.face_colors[facet] = (255, 255, 255, 255)

        scene = cm_out.mesh.scene()
        scene.add_geometry(cm_inp.mesh)
        scene.show(resolution=(400, 400))


    def new_painter(self, image=None):
        """Build a painter to load and process the image"""
        # painter
        self.PaintPanel = painter(self, image)
        self.PaintPanel.close()
        self.stackedWidget.insertWidget(0, self.PaintPanel)
        self.stackedWidget.setCurrentWidget(self.PaintPanel)

    def change_thickness(self, num):
        """Change the width of the painter"""
        self.CurrentWidth = num
        self.PaintPanel.CurrentWidth = num

    def draw_mask(self, maskStype):
        """Draw the mask"""
        self.shape = maskStype
        self.PaintPanel.shape = maskStype

    def clear_mask(self):
        """Clear the mask"""
        self.showImage(self.fname)
        if self.PaintPanel.Brush:
            self.PaintPanel.Brush = False
        else:
            self.PaintPanel.Brush = True

    def set_input(self):
        """Set the input for the network"""
        # get the test mask from painter
        self.PaintPanel.saveDraw()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer.ReadWrite)
        self.PaintPanel.map.save(buffer, 'PNG')
        pil_im = Image.open(io.BytesIO(buffer.data()))

        # transform the image to the tensor
        img = self.transform(self.img)
        value = self.comboBox.currentIndex()
        if True:
            mask = torch.autograd.Variable(self.transform(pil_im)).unsqueeze(0)
            # mask from the random mask
            # mask = Image.open(self.mname)
            # mask = torch.autograd.Variable(self.transform(mask)).unsqueeze(0)
            mask = (mask < 1).float()
        else:
            mask = task.center_mask(img).unsqueeze(0)
        if len(self.opt.gpu_ids) > 0:
            img = img.unsqueeze(0).cuda(self.opt.gpu_ids[0], async=True)
            mask = mask.cuda(self.opt.gpu_ids[0], async=True)

        # get I_m and I_c for image with mask and complement regions for training
        mask = mask
        self.img_truth = img * 2 - 1
        self.img_m = mask * self.img_truth
        self.img_c = (1 - mask) * self.img_truth

        return self.img_m, self.img_c, self.img_truth, mask

    def fill_mask(self):
        """Forward to get the generation results"""
        img_m, img_c, img_truth, mask = self.set_input()

        max_score = 0.0

        results = {'score': [], 'image': []}

        if self.PaintPanel.iteration < 100:

            for i in range(50):

                with torch.no_grad():
                    # encoder process
                    distributions, f = self.model.net_E(img_m)
                    q_distribution = torch.distributions.Normal(distributions[-1][0], distributions[-1][1])
                    #q_distribution = torch.distributions.Normal( torch.zeros_like(distributions[-1][0]), torch.ones_like(distributions[-1][1]))
                    z = q_distribution.sample()

                    # decoder process
                    scale_mask = task.scale_pyramid(mask, 4)
                    img_g, atten = self.model.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask[0].chunk(3, dim=1)[0])
                    img_out = (1 - mask) * img_g[-1].detach() + mask * img_m

                    # get score
                    score = self.model.net_D(img_out).mean()

                    if score > max_score:

                        self.img_g = img_g
                        self.atten = atten
                        self.img_out = img_out

                        max_score = score

                    results['score'].append(score)
                    results['image'].append(util.tensor2im(img_out.detach()))

            # Extract 10 images to show as sample
            samples = [results['image'][i] for i in np.random.randint(0, len(results['score']), 10)]

            for i, s in enumerate(samples):

                new_pil_image = Image.fromarray(s)
                new_qt_image = ImageQt.ImageQt(new_pil_image)
                pixmap = QtGui.QPixmap.fromImage(new_qt_image)
                current_label = getattr(self, f"sample_{i}")
                current_label.setPixmap(pixmap.scaled(178, 178))

            self.label_6.setText(str(round(max_score.item(), 3)))
            self.PaintPanel.iteration += 1

        self.show_result_flag = True
        self.show_result()
