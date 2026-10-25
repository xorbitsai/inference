class RequestManager {
  /**
   * Is 401 being processed?
   */
  private handling401 = false;

  /**
   * Is 403 being processed?
   */
  private handling403 = false;

  canHandle401() {
    if (this.handling401) {
      return false;
    }

    this.handling401 = true;

    return true;
  }

  reset401() {
    this.handling401 = false;
  }

  canHandle403() {
    if (this.handling403) {
      return false;
    }

    this.handling403 = true;

    return true;
  }

  reset403() {
    this.handling403 = false;
  }
}

export const requestManager =
  new RequestManager();