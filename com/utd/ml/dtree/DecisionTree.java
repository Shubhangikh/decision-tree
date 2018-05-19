package com.utd.ml.dtree;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class DecisionTree<T extends Comparable<? super T>> {
    private Node<T> root;

    public DecisionTree() {
        root = new Node<>();
    }

    public Node<T> getRoot() {
        return root;
    }

    public void setRoot(Node<T> root) {
        this.root = root;
    }

    static class Node<T> {
        T data;
        T leftBranchValue;
        T rightBranchValue;
        Node<T> left;
        Node<T> right;
        int label;
        int[] splitProportion;

        public Node() {
            this(null, null, null, null, null, 0, null);
        }
        Node(T data) {
            this(data, null, null, null, null, 0, null);
        }
        Node(T data, T leftBranchValue, T rightBranchValue, Node<T> left, Node<T> right, int label, int[] splitProportion) {
            this.data = data;
            this.leftBranchValue = leftBranchValue;
            this.rightBranchValue = rightBranchValue;
            this.left = left;
            this.right = right;
            this.label = label;
            this.splitProportion = splitProportion;

        }

        public T getData() {
            return data;
        }

        public void setData(T data) {
            this.data = data;
        }

        public Node<T> getLeft() {
            return left;
        }

        public void setLeft(Node<T> left) {
            this.left = left;
        }

        public Node<T> getRight() {
            return right;
        }

        public void setRight(Node<T> right) {
            this.right = right;
        }

        public T getLeftBranchValue() {
            return leftBranchValue;
        }

        public void setLeftBranchValue(T leftBranchValue) {
            this.leftBranchValue = leftBranchValue;
        }

        public T getRightBranchValue() {
            return rightBranchValue;
        }

        public void setRightBranchValue(T rightBranchValue) {
            this.rightBranchValue = rightBranchValue;
        }

        public int getLabel() {
            return label;
        }

        public void setLabel(int label) {
            this.label = label;
        }

        public int[] getSplitProportion() {
            return splitProportion;
        }

        public void setSplitProportion(int[] splitProportion) {
            this.splitProportion = splitProportion;
        }
    }
}
