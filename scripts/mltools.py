class mltools:
    def __init__(self, conf):
        self.lr = conf['LR']
        self.epochs = conf['EPOCHS']
        self.batchsize = conf['BATCHSIZE']
        self.decay = conf['DECAY']
        self.opt = conf['OPTIMIZER']
        self.loss = conf['LOSS']
        self.mets = conf['METRICS']
        self.moment = conf['MOMENTUM']
        self.nesterov = conf['NESTEROV']
        self.summaryflag = conf['SUMMARY_FLAG']


    def cnn_baseline_v1(self):
        """
        Baseline CNN model without taking incidence angle into consideration
        :return:
        """
        from keras.models import Input, Model
        from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

        mdlin = Input(shape=(75, 75, 2), name='CNN-Input')

        x1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', name='Conv-1')(mdlin)
        x1 = Dropout(0.3, name='Drop-1')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), name='Pool-1')(x1)
        x1 = Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu', name='Conv-2')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), name='Pool-2')(x1)
        x1 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu', name='Conv-3')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same', name='Pool-3')(x1)
        x1 = Flatten(name='Flatten-1')(x1)
        x1 = Dense(128, activation='relu', name='Dense-2')(x1)

        mdlout = Dense(2, activation='relu', name='Model-Output')(x1)

        model = Model(inputs=mdlin, outputs=mdlout)

        opt = self.getopt()
        model.compile(optimizer=opt, loss=self.loss)

        if self.summaryflag:
            model.summary()

        return model


    def cnn_split_band_v1(self):
        """
        Baseline CNN model without taking incidence angle into consideration
        :return:
        """
        from keras.models import Input, Model
        from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

        mdlin = Input(shape=(75, 75, 1), name='CNN-Input')

        x1 = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', name='Conv-1')(mdlin)
        x1 = Dropout(0.3, name='Drop-1')(x1)
        x1 = MaxPooling2D(pool_size=(2,2), name='Pool-1')(x1)
        x1 = Conv2D(filters=64, kernel_size=(4,4), padding='same', activation='relu', name='Conv-2')(x1)
        x1 = MaxPooling2D(pool_size=(2,2), name='Pool-2')(x1)
        x1 = Conv2D(filters=128, kernel_size=(2,2), activation='relu', name='Conv-3')(x1)
        x1 = MaxPooling2D(pool_size=(2,2), padding='same', name='Pool-3')(x1)
        x1 = Flatten(name='Flatten-1')(x1)
        x1 = Dense(128, activation='relu', name='Dense-2')(x1)

        mdlout = Dense(2, activation='relu', name='Model-Output')(x1)

        model = Model(inputs=mdlin, outputs=mdlout)

        opt = self.getopt()
        model.compile(optimizer=opt, loss=self.loss)

        if self.summaryflag:
            model.summary()

        return model


    def predang_cnn_v1(self):
        """
        Baseline CNN model for predicting the incidence angle, based on SAR image(s).
        :return: Keras model object.
        """
        from keras.models import Model
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

        mdlin = Input(shape=(75, 75, 2), name='CNN-Input')

        x1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='Conv-1')(mdlin)
        x1 = Dropout(0.3, name='Drop-1')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), name='Pool-1')(x1)
        x1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='Conv-2')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), name='Pool-2')(x1)
        x1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='Conv-3')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same', name='Pool-4')(x1)
        x1 = Flatten(name='Flatten-1')(x1)
        x1 = Dense(128, activation='relu', name='Dense-2')(x1)

        mdlout = Dense(1, activation='relu', name='Model-Output')(x1)

        model = Model(inputs=mdlin, outputs=mdlout)

        opt = self.getopt()
        model.compile(optimizer=opt, loss=self.loss)

        if self.summaryflag:
            model.summary()

        return model


    def train(self, mdl):
        """
        Train a compiled Keras model.
        :param mdl: Keras model object.
        :return: Keras history object.
        """
        mdlhist = mdl.fit(x=self.x_train,
                          y=self.y_train,
                          validation_data=(self.x_val, self.y_val),
                          batch_size=self.batchsize,
                          epochs=self.epochs)

        return mdlhist


    def predict(self, mdl, testdat):
        """
        Predict output from Keras model.
        :param model: Keras model object.
        :param testdat: Test data to predict on.
        :return: Numpy array with predictions.
        """
        predicts = mdl.predict(x=testdat,
                                 batch_size=self.batchsize,
                                 verbose=1)

        return predicts


    def getopt(self):
        """
        Get Keras optimizer object from specified optimizer within the object CONF.
        :return: Keras optimizer object.
        """
        import sys
        from keras import optimizers

        if self.opt.upper() == 'SGD':
            opt = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.moment, nesterov=self.nesterov)

        elif self.opt.upper() == 'ADAM':
            opt = optimizers.Adam(lr=self.lr, decay=self.decay)

        else:
            sys.exit()

        return opt