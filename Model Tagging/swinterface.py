class FeatureTag:
    def __init__(self, app, features):
        self.app = app
        self.colors = []
        self.colordict = []
        self.features = features[:]
        self.features.insert(0, "No Feature")
        self.currFeature = 0

    def set_colors(self, colors):
        self.colors = colors[:]
        self.colors.insert(0, (0.792156862745098, 0.8196078431372549, 0.9333333333333333))

        self.colordict = dict(zip(self.features, self.colors))

    def set_feature(self, feature):
        self.currFeature = feature

    def update_label(self):
        if self.colors is None or self.features is None:
            return False

        v_face_prop = self.app.ActiveDoc.MaterialPropertyValues
        rgbcolor = self.colordict[self.currFeature]

        tmp = []
        for c in rgbcolor:
            tmp.append(int(c*255))

        rgbcolor = tuple(tmp)

        color = ''.join('{:02X}'.format(c) for c in rgbcolor[::-1])
        color = int(color, base=16)
        self.app.ActiveDoc.SelectedFaceProperties(color, v_face_prop[3], v_face_prop[4],
                                                  v_face_prop[5], v_face_prop[6],
                                                  v_face_prop[7], v_face_prop[8], False,
                                                  self.currFeature)
        success = self.app.ActiveDoc.SelectionManager.GetSelectedObject6(1, -1).DeSelect
