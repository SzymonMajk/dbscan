import numpy as np
from scipy.spatial import distance


class Component:

  def __init__(self, index, coords):
    self.index = index
    self.coords = coords
    self.label = -1
    self.point_type = "new"
    self.neighboors = []
    self.neighboors_indexes = {index}

  def add_if_neighboor(self, other, eps, metric, p):
    dist = 0.0

    if metric == "euclidean":
      dist = distance.euclidean(self.coords, other.coords)
    elif metric == "minkowski":
      dist = distance.minkowski(self.coords, other.coords, p)
    elif metric == "chebyshev":
      dist = distance.chebyshev(self.coords, other.coords)
    elif metric == "cityblock":
      dist = distance.cityblock(self.coords, other.coords)
    else:
      dist = distance.euclidean(self.coords, other.coords)

    if dist <= eps and other.index not in self.neighboors_indexes:
      self.neighboors_indexes.add(other.index)
      self.neighboors.append(other)

  def set_type(self, min_samples):
    if len(self.neighboors) >= min_samples:
      self.point_type = "core"
    elif self.label == -1:
      self.point_type = "outlier"
    else:
      self.point_type = "border"


class Dbscan:
  def __init__(self, eps=0.25, min_samples=12, metric="euclidean", p=3):
    self.eps = eps
    self.min_samples = min_samples - 1
    self.metric = metric
    self.p = p
    self.components_ = np.empty(0)
    self.components = []
    self.current_label = 0

  def fit(self, data):
    self.components_ = data
    self.components = []

    for i in range(len(self.components_)):
      self.components.append(Component(i, self.components_[i]))

    for c in self.components:
      if c.point_type is "new":
        self.start_expand(c, self.components)

    self.labels_ = np.array([c.label for c in self.components])
    self.core_sample_indices_ = np.array([c.index for c in self.components if c.point_type is "core"])
    return self

  def partial_fit(self, data):
    self.components_ = np.append(self.components, data)
    old_components_count = len(self.components)

    for i in range(len(data)):
      self.components.append(Component(i + old_components_count, data[i]))

    for c in self.components:
      if c.point_type is "core":
        self.continue_expand(c, self.components, c.label)
      elif c.point_type is "border":
        self.continue_expand(c, self.components, c.label)

    for c in self.components:
      if c.point_type is "new":
        self.start_expand(c, self.components)

    self.labels_ = np.array([c.label for c in self.components])
    self.core_sample_indices_ = np.array([c.index for c in self.components if c.point_type is "core"])

    return self

  def start_expand(self, component, components):
    self.find_nearest(component, components)
    component.set_type(self.min_samples)

    if component.point_type is "core":
      component.label = self.current_label
      for n in component.neighboors:
        self.continue_expand(n, components, component.label)
      self.current_label += 1

  def continue_expand(self, component, components, label):
    self.find_nearest(component, components)
    component.label = label
    component.set_type(self.min_samples)

    if component.point_type is "core":
      for n in component.neighboors:
        if (n.label == -1):
          self.continue_expand(n, components, component.label)

  def find_nearest(self, component, components):
    for c in components:
      component.add_if_neighboor(c, self.eps, self.metric, self.p)