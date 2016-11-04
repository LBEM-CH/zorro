# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_ViewDialog.ui'
#
# Created: Mon Oct 24 17:17:36 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_ViewDialog(object):
    def setupUi(self, ViewDialog):
        ViewDialog.setObjectName("ViewDialog")
        ViewDialog.resize(640, 480)
        self.gridLayout = QtGui.QGridLayout(ViewDialog)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.view = ViewWidget(ViewDialog)
        self.view.setMaximumSize(QtCore.QSize(16777214, 16777214))
        self.view.setObjectName("view")
        self.gridLayout.addWidget(self.view, 0, 0, 1, 1)

        self.retranslateUi(ViewDialog)
        QtCore.QMetaObject.connectSlotsByName(ViewDialog)

    def retranslateUi(self, ViewDialog):
        ViewDialog.setWindowTitle(QtGui.QApplication.translate("ViewDialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))

from .ViewWidget import ViewWidget
